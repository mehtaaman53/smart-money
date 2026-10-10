import json, requests
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36","Referer":"https://www.bseindia.com/","Origin":"https://www.bseindia.com","Accept":"application/json, text/plain, */*"}
out={}
for name,url in [("scrips","https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active"),
                 ("industries","https://api.bseindia.com/BseIndiaAPI/api/IndustryMaster/w")]:
    try:
        r=requests.get(url,headers=UA,timeout=40)
        try:
            j=r.json(); n=len(j) if isinstance(j,list) else len(j.get("Table",[])) if isinstance(j,dict) else None
            sample=(j[:2] if isinstance(j,list) else j)
            out[name]={"status":r.status_code,"n":n,"sample":json.dumps(sample)[:1200]}
        except Exception:
            out[name]={"status":r.status_code,"text":r.text[:300]}
    except Exception as e:
        out[name]={"err":str(e)[:200]}
print(json.dumps(out,indent=1)); json.dump(out,open("probe/bse_result.json","w"),indent=1)
