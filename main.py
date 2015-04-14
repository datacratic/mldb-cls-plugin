
import urllib, datetime, csv


######################
##  Create toy dataset
######################
mldb.perform("DELETE", "/v1/datasets/toy", [], {})

# create a mutable beh dataset
datasetConfig = {
        "type": "mutable",
        "id": "toy",
    }

dataset = mldb.create_dataset(datasetConfig)

def featProc(k, v):
    if k=="Pclass": return "c"+v
    if k=="Cabin": return v[0]
    return v

ts = datetime.datetime.now()
titanic_dataset = "https://raw.githubusercontent.com/datacratic/mldb-pytanic-plugin/master/titanic_train.csv"
for idx, csvLine in enumerate(csv.DictReader(urllib.urlopen(titanic_dataset))):
    tuples = [[k,featProc(k,v),ts] for k,v in csvLine.iteritems() if k != "PassengerId" and v!=""]
    dataset.record_row(csvLine["PassengerId"], tuples)

# commit the dataset
dataset.commit()




######################
##  Handle custom routes
######################
def requestHandler(mldb, remaining, verb, resource, restParams, payload, contentType, contentLength, headers):
    print "Handling route in python"
    import json, re

    def isNumber(x):
        try:
            val = float(x)
            return True
        except:
            return False

    # http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python
    def _decode_list(data):
        rv = []
        for item in data:
            if isinstance(item, unicode):
                item = item.encode('utf-8')
            elif isinstance(item, list):
                item = _decode_list(item)
            elif isinstance(item, dict):
                item = _decode_dict(item)
            rv.append(item)
        return rv

    def _decode_dict(data):
        rv = {}
        for key, value in data.iteritems():
            if isinstance(key, unicode):
                key = key.encode('utf-8')
            if isinstance(value, unicode):
                value = value.encode('utf-8')
            elif isinstance(value, list):
                value = _decode_list(value)
            elif isinstance(value, dict):
                value = _decode_dict(value)
            rv[key] = value
        return rv

    if verb == "GET" and remaining == "/dataset-details":
        datasets = []
        rez = mldb.perform("GET", "/v1/datasets", [], {})
        for dataset in json.loads(rez["response"]):
            rez = mldb.perform("GET", str("/v1/datasets/%s" % dataset), [], {})
            resp = json.loads(rez["response"])

            datasets.append({
                "id": str(dataset),
                "type": str(resp["type"]),
                "rowCount": resp["status"]["rowCount"],
                "valueCount": resp["status"]["valueCount"]
            })
        
        return datasets
    
    if verb == "GET" and remaining == "/classifier-list":
        rez = mldb.perform("GET", "/v1/pipelines", [], {})
        pipelines = []
        for pipeline in json.loads(rez["response"]):
            # get the piepline details
            rez = mldb.perform("GET", str("/v1/pipelines/%s" % pipeline), [], {})
            resp = json.loads(rez["response"])
            if resp["type"] != "classifier": continue
            
            # get the runs for the piepeline
            rez_runs = mldb.perform("GET", str("/v1/pipelines/%s/runs" % pipeline), [], {})
            resp_runs = json.loads(rez_runs["response"])

            resp_last_run = {}
            if len(resp_runs) > 0:
                rez_last_run = mldb.perform("GET", str("/v1/pipelines/%s/runs/%s" % (pipeline, resp_runs[-1])), [], {})
                resp_last_run = json.loads(rez_last_run["response"])

            pipelines.append({
                "id": str(pipeline),
                "state": str(resp["state"]),
                "type": str(resp["type"]),
                "params": _decode_dict(resp["config"]["params"]),
                "runs": _decode_list(resp_runs),
                "last_run": _decode_dict(resp_last_run)
            })
        return pipelines

    if verb == "GET" and remaining == "/cls-presets":
        lines2 = [x.strip() for x in open(mldb.plugin.get_plugin_dir() +"/classifier-config.txt") if len(x.strip())>0 and x[0] != "#"]


        reobj = re.compile(r"([\w]+) \{", re.IGNORECASE)

        configs = {}
        accum = ""
        num_open_brackets = 0
        for l in lines2:
            # if we're done
            if l[0] == "}" and num_open_brackets == 1:
                accum += l.replace(";", "")
                parsedConf = json.loads(str("{"+accum+"}"), object_hook=_decode_dict)

                configs[parsedConf.keys()[0]] = parsedConf.values()[0]

                accum = ""
                num_open_brackets = 0
                continue

            if "{" in l:
                num_open_brackets += 1
            if "}" in l:
                num_open_brackets -= 1

            if len(accum)>1 and l[0] != "}" and (accum[-1] == '"' or isNumber(accum[-1]) or accum[-1] == '}'):
                accum += ","

            sl = l.split("=")
            if len(sl) > 1:
                key = '"' + sl[0] + '"'
                val = sl[1].replace(";", "");
                if isNumber(val):
                    val = str(val)
                else:
                    val = '"' + val + '"'

                accum += key + ":" + val
            else:
                match = reobj.search(l)
                if match:
                    accum += '"'+match.group(1)+'": {'
                else:
                    accum += l.replace(";", "")

        print configs
        return configs

mldb.plugin.set_request_handler(requestHandler)


######################
##  Create route to serve html
######################
mldb.plugin.serve_static_folder("static", "static")

