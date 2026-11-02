"""Pump relay drivers. SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM.

A relay driver mirrors the pump permit onto a relay. It is driven by permit.PermitOutput and nothing else:
cut() when the interlock says permit OFF, restore() only after PermitOutput redeemed a ResetGrant. The model
diagnoser has no path here (a static test checks that only permit.py calls cut/restore).

ScreenRelay is the default: it only records the state so the dashboard can show it.

UNTESTED HARDWARE: UsbSerialRelay has never been run against a real relay board. The byte frames are the
common LCUS-style USB relay protocol and are an assumption. It needs pyserial, which is NOT a project
dependency. Tests use a fake serial object only. Do not rely on it for anything real without bench testing.
"""


class RelayDriver:
    """Interface. cut(): open the relay (pump cannot run). restore(): close it again (re-arm)."""

    driver = "base"

    def cut(self) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict:
        raise NotImplementedError


class ScreenRelay(RelayDriver):
    driver = "screen"

    def __init__(self):
        self.pump_enabled = True

    def cut(self) -> None:
        self.pump_enabled = False

    def restore(self) -> None:
        self.pump_enabled = True

    def snapshot(self) -> dict:
        return {"driver": self.driver, "pump_enabled": self.pump_enabled}


class UsbSerialRelay(RelayDriver):
    """UNTESTED HARDWARE path. `opener(port)` returns an object with write(bytes) and close(); the default
    opens a pyserial port at 9600 baud. Opening sends nothing, so building one never switches the relay."""

    driver = "usb-serial"
    UNTESTED_HARDWARE = True
    OFF_FRAME = bytes([0xA0, 0x01, 0x00, 0xA1])  # channel 1 open
    ON_FRAME = bytes([0xA0, 0x01, 0x01, 0xA2])  # channel 1 closed

    def __init__(self, port: str, opener=None):
        self.port, self.pump_enabled, self.fault = port, True, False
        self._ser = (opener or self._open_pyserial)(port)

    @staticmethod
    def _open_pyserial(port):
        import serial  # pyserial: not a project dependency

        return serial.Serial(port, 9600, timeout=1)

    def _send(self, frame: bytes) -> None:
        try:
            self._ser.write(frame)
        except Exception:
            self.fault = True
            raise

    def cut(self) -> None:
        self.pump_enabled = False  # the intent is recorded even if the write then fails
        self._send(self.OFF_FRAME)

    def restore(self) -> None:
        self._send(self.ON_FRAME)
        self.pump_enabled = True

    def snapshot(self) -> dict:
        return {"driver": self.driver, "pump_enabled": self.pump_enabled, "fault": self.fault,
                "untested_hardware": True}

    def close(self) -> None:
        self._ser.close()
