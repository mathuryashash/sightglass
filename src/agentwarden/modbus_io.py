"""Modbus/TCP tag map on 127.0.0.1 only (SIMULATED controller; never real equipment).

Unit 1 = PLC:    holding reg 0 = reported level x10, coil 0 = pump_running, coil 1 = pump_permit.
Unit 2 = camera: holding reg 0 = level x10, reg 1 = confidence x100, reg 2 = status code.
pump_permit is written ONLY by the interlock, via ModbusClient.write_permit (called from runner).
The tag values live in a plain in-process TagStore; pymodbus serves them through SimDevice actions.
"""
import asyncio
import socket
import threading

from pymodbus.client import ModbusTcpClient
from pymodbus.constants import ExcCodes
from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice

UNIT_PLC, UNIT_CAMERA = 1, 2
REG_LEVEL, REG_CONF, REG_STATUS = 0, 1, 2
COIL_PUMP_RUNNING, COIL_PERMIT = 0, 1
STATUS_CODES = {"OK": 0, "EMPTY": 1, "CAMERA_BLIND": 2, "LOW_CONFIDENCE": 3}
STATUS_NAMES = {v: k for k, v in STATUS_CODES.items()}
NO_LEVEL = 0xFFFF  # camera level sentinel: no reading
LOCAL_HOSTS = ("127.0.0.1", "localhost")
FC_WRITE_COIL, FC_WRITE_COILS = 5, 15
COIL_FCS = (1, 2, FC_WRITE_COIL, FC_WRITE_COILS)


def _check_host(host: str) -> str:
    if host not in LOCAL_HOSTS:
        raise ValueError(f"modbus is bound to 127.0.0.1 only, refusing host {host!r}")
    return "127.0.0.1"


def _x10(pct: float) -> int:
    return int(round(min(100.0, max(0.0, pct)) * 10))


class TagStore:
    def __init__(self):
        self.plc_level_x10 = 0
        self.pump_running = False
        self.permit = True  # starts permitted; only the interlock path clears it
        self.camera = [NO_LEVEL, 0, STATUS_CODES["CAMERA_BLIND"]]


def _blocks():
    bits = [SimData(0, count=8, values=False, datatype=DataType.BITS)]
    regs = [SimData(0, count=8, values=0, datatype=DataType.REGISTERS)]
    return (bits, list(bits), regs, list(regs))  # coils, discrete, holding, input


def _devices(store: TagStore) -> list[SimDevice]:
    # pymodbus hands the action the whole block (current[0] is address `start`); coils are bit-packed
    # into register 0 (bit 0 = pump_running, bit 1 = pump_permit).
    async def plc_action(fc, start, address, count, current, set_values):
        if fc in COIL_FCS:
            if set_values is not None:  # the only writable tag is the permit coil
                if fc not in (FC_WRITE_COIL, FC_WRITE_COILS) or not (
                    address <= COIL_PERMIT < address + len(set_values)
                ):
                    return ExcCodes.ILLEGAL_ADDRESS
                store.permit = bool(set_values[COIL_PERMIT - address])
            current[0] = int(store.pump_running) | (int(store.permit) << COIL_PERMIT)
            return None
        if set_values is not None:
            return ExcCodes.ILLEGAL_ADDRESS
        current[REG_LEVEL] = store.plc_level_x10
        return None

    async def camera_action(fc, start, address, count, current, set_values):
        if set_values is not None:
            return ExcCodes.ILLEGAL_ADDRESS  # camera tag is read-only over the wire
        current[: len(store.camera)] = store.camera
        return None

    return [
        SimDevice(UNIT_PLC, simdata=_blocks(), action=plc_action),
        SimDevice(UNIT_CAMERA, simdata=_blocks(), action=camera_action),
    ]


class ModbusPlcServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = _check_host(host)
        self.port = port
        self._store = TagStore()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: ModbusTcpServer | None = None

    # -- lifecycle
    def start(self) -> "ModbusPlcServer":
        if self.port == 0:
            with socket.socket() as s:
                s.bind((self.host, 0))
                self.port = s.getsockname()[1]
        ready, errors = threading.Event(), []

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            async def boot():  # the server needs a running loop at construction
                self._server = ModbusTcpServer(_devices(self._store), address=(self.host, self.port))
                await self._server.listen()

            try:
                self._loop.run_until_complete(boot())
            except Exception as e:  # surface bind failures to the caller
                errors.append(e)
                ready.set()
                return
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(10)
        if errors:
            raise errors[0]
        return self

    def stop(self) -> None:
        if self._loop and self._server and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._server.shutdown(), self._loop).result(10)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(10)
            self._thread = None

    def __enter__(self) -> "ModbusPlcServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- server-side tag access (the simulated plant / camera publish here)
    def publish_plc(self, level_pct: float, pump_running: bool) -> None:
        self._store.plc_level_x10 = _x10(level_pct)
        self._store.pump_running = bool(pump_running)

    def publish_camera(self, level_pct: float | None, confidence: float, status: str) -> None:
        level = NO_LEVEL if level_pct is None else _x10(level_pct)
        conf = int(round(min(1.0, max(0.0, confidence)) * 100))
        self._store.camera = [level, conf, STATUS_CODES[status]]

    @property
    def permit(self) -> bool:
        return self._store.permit


class ModbusClient:
    """Small typed wrapper over a pymodbus TCP client."""

    def __init__(self, host: str, port: int):
        self._host = _check_host(host)
        self._c = ModbusTcpClient(self._host, port=port)
        if not self._c.connect():
            raise ConnectionError(f"cannot connect to modbus at {self._host}:{port}")

    def close(self) -> None:
        self._c.close()

    def _regs(self, unit: int, count: int) -> list[int]:
        r = self._c.read_holding_registers(0, count=count, device_id=unit)
        if r.isError():
            raise IOError(f"modbus read failed: {r}")
        return list(r.registers)

    def _coil(self, address: int) -> bool:
        r = self._c.read_coils(address, count=1, device_id=UNIT_PLC)
        if r.isError():
            raise IOError(f"modbus read failed: {r}")
        return bool(r.bits[0])

    def read_plc_level(self) -> float:
        return self._regs(UNIT_PLC, 1)[REG_LEVEL] / 10

    def read_pump_running(self) -> bool:
        return self._coil(COIL_PUMP_RUNNING)

    def read_permit(self) -> bool:
        return self._coil(COIL_PERMIT)

    def read_camera(self) -> tuple[float | None, float, str]:
        level, conf, status = self._regs(UNIT_CAMERA, 3)
        return (None if level == NO_LEVEL else level / 10), conf / 100, STATUS_NAMES[status]

    def write_permit(self, permit: bool) -> None:
        r = self._c.write_coil(COIL_PERMIT, permit, device_id=UNIT_PLC)
        if r.isError():
            raise IOError(f"modbus write failed: {r}")
