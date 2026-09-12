import asyncio
from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics

def det(node_id, text, status=Status.COMPLETED):
    ar = AgentResult(stop_reason="end_turn", message={"role":"assistant","content":[{"text":text}]},
                     metrics=EventLoopMetrics(), state={})
    return MultiAgentResult(status=status, results={node_id: NodeResult(result=ar, status=status)})

ATTEMPTS = {"n": 0}

class Extract(MultiAgentBase):
    def __init__(s): super().__init__(); s.id="extract"
    async def invoke_async(s, task, invocation_state=None, **kw):
        ATTEMPTS["n"] += 1
        print(f"  [extract] attempt {ATTEMPTS['n']}")
        return det(s.id, f"attempt={ATTEMPTS['n']}")

class Reconcile(MultiAgentBase):
    def __init__(s): super().__init__(); s.id="reconcile"
    async def invoke_async(s, task, invocation_state=None, **kw):
        ok = ATTEMPTS["n"] >= 3     # reconciles only on 3rd extraction
        print(f"  [reconcile] reconciled={ok}")
        return det(s.id, "RECONCILED" if ok else "MISMATCH")

class Audit(MultiAgentBase):
    def __init__(s): super().__init__(); s.id="audit"
    async def invoke_async(s, task, invocation_state=None, **kw):
        print("  [audit] running 4.26"); return det(s.id, "80%")

def reconciled(state):
    r = state.results.get("reconcile")
    return r is not None and "RECONCILED" in str(r)
def not_reconciled(state):
    r = state.results.get("reconcile")
    return r is not None and "MISMATCH" in str(r) and ATTEMPTS["n"] < 4

b = GraphBuilder()
b.add_node(Extract(), "extract"); b.add_node(Reconcile(), "reconcile"); b.add_node(Audit(), "audit")
b.add_edge("extract","reconcile")
b.add_edge("reconcile","extract", condition=not_reconciled)   # THE CYCLE
b.add_edge("reconcile","audit", condition=reconciled)
b.set_entry_point("extract"); b.reset_on_revisit(True); b.set_max_node_executions(12)
g = b.build()
r = asyncio.run(g.invoke_async("letter"))
print("STATUS:", r.status)
print("execution_order:", [n.node_id for n in r.execution_order])
