"""Adversarial self-check mirroring the strict review. Run before submitting:
    PYTHONPATH=src python tools_selfcheck.py
Exits non-zero if any blocker/hardening check fails.
"""
import os, sys
import pickup_putdown.evaluation as ev
E, P, Ig, C = ev.EvaluationEvent, ev.EvaluationPrediction, ev.EvaluationIgnoreInterval, ev.Criterion
fails = []
def chk(n, c):
    print(("PASS" if c else "FAIL") + "  " + n)
    if not c: fails.append(n)
def raises(f):
    try: f(); return False
    except Exception: return True

g = [E("c","pickup",1.3,3.8,"g1"), E("c","pickup",2.8,5.3,"g2")]
p = [P("c","pickup",1.0,2.0,"p1",score=0.6), P("c","pickup",2.2,4.2,"p2",score=0.9)]
chk("mAP order-invariant", ev.average_precision(g,p,"pickup",0.5)==ev.average_precision(list(reversed(g)),list(reversed(p)),"pickup",0.5))
r = ev.match_one_to_one(g,p,C("tiou",0.5)); chk("matcher adversarial 1,1,1", (r.tp,r.fp,r.fn)==(1,1,1))
m = ev.aggregate_metrics([E("c","pickup",1.0,2.0,"g1")],[],{"c":10.0},ignores=[Ig("c",0.0,5.0)])
chk("ignore-consistent event count", m["event_count_error_per_clip"]==0 and m["event_count_error_absolute"]==0)
chk("overlap not multi by default", ev.aggregate_metrics([E("c","pickup",0.0,2.0,"g1"),E("c","pickup",1.0,3.0,"g2")],[P("c","pickup",0.0,2.0,"p1")],{"c":100.0})["multi_item_recall"] is None)
class Min:
    def __init__(s,c,t,a,b): s.clip_id,s.type,s.t_start,s.t_end=c,t,a,b
try: ev.aggregate_metrics([Min("c","pickup",1.0,2.0)],[Min("c","pickup",1.0,2.0)],{"c":10.0}); ev.slice_metrics([Min("c","pickup",1.0,2.0)],[Min("c","pickup",1.0,2.0)],{"c":10.0}); ev.failure_gallery([Min("c","pickup",1.0,2.0)],[Min("c","pickup",1.0,2.0)],C("tiou",0.5)); mok=True
except Exception as e: mok=False; print("   crash:",repr(e))
chk("minimal objects don't crash", mok)
chk("score 0.0 preserved", ev.predictions_from_rows([{"clip_id":"c","type":"pickup","t_start":"1","t_end":"2","score":"0"}])[0].score==0.0)
chk("invalid criterion rejected", raises(lambda: C("bogus",0.5)) and raises(lambda: C("tiou",1.5)))
chk("negative timestamp rejected", raises(lambda: P("c","pickup",-1.0,2.0)))
chk("invalid ignore rejected", raises(lambda: Ig("c",5.0,1.0)))
chk("negative clip duration handled", raises(lambda: ev.aggregate_metrics([E("c","pickup",1.0,2.0)],[],{"c":-5.0})))
chk("HTML escaped", "&lt;s&gt;" in ev.render_html(ev.aggregate_metrics([E("c","pickup",1.0,2.0)],[P("c","pickup",1.0,2.0)],{"c":10.0}),"<s>"))
chk("acceptance config present", os.path.exists("configs/evaluation_acceptance.yaml"))
chk("sample output present", os.path.exists("samples/sample_metrics.json"))
chk("scipy declared", os.path.exists("pyproject.toml") and os.path.exists("requirements.txt"))
print(("\nALL CLEAR" if not fails else f"\n{len(fails)} FAILED: "+", ".join(fails)))
sys.exit(1 if fails else 0)
