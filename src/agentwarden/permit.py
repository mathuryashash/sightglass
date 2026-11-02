"""The only module that touches the pump-permit coil. SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM.

apply(actions) lowers the permit when the interlock says WRITE_PERMIT_FALSE. rearm(grant) raises it, and only
with a ResetGrant that interlock.reset_with_grant issued for an accepted human reset. Nothing else here (or
anywhere on the wire) can make the permit True.

The pump relay (relay.py) follows the same two moves and nothing else: cut() on a trip, restore() only after the
grant was redeemed. A relay that fails never undoes the software cut (relay_fault is set instead).
"""
import logging

from agentwarden import interlock
from agentwarden.relay import ScreenRelay

log = logging.getLogger(__name__)


class PermitOutput:
    def __init__(self, server=None, client=None, relay=None):
        self._server, self._client = server, client
        self._permit = True  # starts permitted, like the tag store
        self.relay = relay if relay is not None else ScreenRelay()
        self.relay_fault = False

    @property
    def permit(self) -> bool:
        return self._client.read_permit() if self._client else self._permit

    def apply(self, actions) -> None:
        if interlock.Action.WRITE_PERMIT_FALSE in actions:
            self._permit = False
            try:
                self.relay.cut()
            except Exception as e:  # the software cut stands; the dashboard shows the relay fault
                self.relay_fault = True
                log.error("relay cut failed: %s", type(e).__name__)
            if self._client:
                self._client.clear_permit()

    def rearm(self, grant) -> None:
        if self._server:
            self._server.rearm_from_reset(grant)  # redeems the grant
        else:
            interlock.redeem(grant)
        self._permit = True
        try:
            self.relay.restore()
        except Exception as e:  # relay stays open: the safe direction
            self.relay_fault = True
            log.error("relay restore failed: %s", type(e).__name__)
