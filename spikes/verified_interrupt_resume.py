import asyncio, sys, json, os
from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status
from strands.interrupt import Interrupt
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics

def det_result(node_id, text, status=Status.COMPLETED):
    ar = AgentResult(stop_reason="end_turn",
                     message={"role":"assistant","content":[{"text":text}]},
                     metrics=EventLoopMetrics(), state={})
    return MultiAgentResult(status=status, results={node_id: NodeResult(result=ar, status=status)})
from strands.session.file_session_manager import FileSessionManager
from strands.storage import LocalFileStorage

BASE = os.path.dirname(os.path.abspath(__file__))

class DetNode(MultiAgentBase):
    """Deterministic node. No model. Raises its own interrupt."""
    def __init__(self, node_id):
        super().__init__()
        self.id = node_id
    async def invoke_async(self, task, invocation_state=None, **kw):
        print(f"  [{self.id}] invoked with task type={type(task).__name__} repr={str(task)[:200]}")
        # detect resume: task is a list of interruptResponse blocks
        resumed = None
        if isinstance(task, list):
            for b in task:
                if isinstance(b, dict) and "interruptResponse" in b:
                    resumed = b["interruptResponse"]["response"]
        if resumed is None:
            print(f"  [{self.id}] RAISING INTERRUPT (deterministic, zero model calls)")
            itp = Interrupt(id="bilateral-q1", name="confirm_bilateral",
                            reason={"branch_a": 80, "branch_b": 70, "delta_usd": 285})
            return MultiAgentResult(status=Status.INTERRUPTED, results={}, interrupts=[itp])
        print(f"  [{self.id}] RESUMED with human answer = {resumed!r} -> computing")
        return det_result(self.id, f"combined=80 (human said {resumed})")

class ReportNode(MultiAgentBase):
    def __init__(self, node_id):
        super().__init__(); self.id = node_id
    async def invoke_async(self, task, invocation_state=None, **kw):
        print(f"  [{self.id}] writing report")
        return det_result(self.id, "REPORT WRITTEN")

def build():
    sm = FileSessionManager(session_id="spike-1", storage_dir=os.path.join(BASE,".strands"))
    b = GraphBuilder()
    b.add_node(DetNode("engine"), "engine")
    b.add_node(ReportNode("report"), "report")
    b.add_edge("engine", "report")
    b.set_entry_point("engine")
    b.set_session_manager(sm)
    return b.build()

async def main():
    phase = sys.argv[1]
    g = build()
    if phase == "run":
        r = await g.invoke_async("audit letter")
        print("STATUS:", r.status, "| interrupts:", [(i.id,i.name,i.reason) for i in r.interrupts])
        print("execution_order:", [n.node_id for n in r.execution_order])
        with open(os.path.join(BASE,"gstate.json"),"w") as f: json.dump(g.serialize_state(), f)
        print("serialized graph state to gstate.json")
    else:
        with open(os.path.join(BASE,"gstate.json")) as f: g.deserialize_state(json.load(f))
        print("deserialized. resuming...")
        r = await g.invoke_async([{"interruptResponse":{"interruptId":"bilateral-q1","response":"yes"}}])
        print("STATUS:", r.status)
        print("execution_order:", [n.node_id for n in r.execution_order])
        for k,v in r.results.items(): print("  result",k,"=",v.result)

asyncio.run(main())
