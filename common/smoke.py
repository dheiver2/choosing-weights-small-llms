import time,sys,os; sys.path.insert(0,os.path.dirname(__file__))
import evalkit as E
for name in sys.argv[1:]:
    t=time.time(); m,tok=E.load(f'models/{name}'); print(name,'load',round(time.time()-t,1),'s',flush=True)
    for task in E.TASKS:
        t=time.time(); r=E.eval_task(m,tok,task,'val'); print(' ',task,'val acc',round(r.mean(),3),'n',len(r),round(time.time()-t,1),'s',flush=True)
    t=time.time(); print('  wikitext nll',E.wikitext_nll(m,tok).mean().round(3),round(time.time()-t,1),'s',flush=True)
    del m
