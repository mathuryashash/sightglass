"""The only module that touches the pump-permit coil. SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM.

apply(actions) lowers the permit when the interlock says WRITE_PERMIT_FALSE. rearm(grant) raises it, and only
with a ResetGrant that interlock.reset_with_grant issued for an accepted human reset. Nothing else here (or
anywhere on the wire) can make the permit True.
"""
from agentwarden import interlock


class PermitOutput:
    def __init__(self, server=None, client=None):
        self._server, self._client = server, client
        self._permit = True  # starts permitted, like the tag store

    @property
    def permit(self) -> bool:
        return self._client.read_permit() if self._client else self._permit

    def apply(self, actions) -> None:
        if interlock.Action.WRITE_PERMIT_FALSE in actions:
            self._permit = False
            if self._client:
                self._client.clear_permit()

    def rearm(self, grant) -> None:
        if self._server:
            self._server.rearm_from_reset(grant)  # redeems the grant
        else:
            interlock.redeem(grant)
        self._permit = True
