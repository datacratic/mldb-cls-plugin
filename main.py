
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

            # skip the datasets we generate using the cls plugin
            if dataset.startswith("cls-plugin-"): continue

            datasets.append({
                "id": str(dataset),
                "type": str(resp["type"]),
                "rowCount": resp["status"]["rowCount"],
                "valueCount": resp["status"]["valueCount"]
            })
        
        return datasets
   
    def get_accuracy_pipeline_name(pipeline, run):
        return "cls-plugin-%s-%s" % (pipeline, run)
     
    if verb == "GET" and remaining.startswith("/cls-details"):
        pipeline_name = remaining.split("/")[-1]
        rez = mldb.perform("GET", "/v1/pipelines/"+pipeline_name, [], {})
        prez = json.loads(rez["response"])
        pipeline_details = _decode_dict(prez)
            
        rez_runs = mldb.perform("GET", str("/v1/pipelines/%s/runs" % pipeline_name), [], {})
        resp_runs = json.loads(rez_runs["response"])

        resp_all_run = []
        for run_id in resp_runs:
            rez_all_run = mldb.perform("GET", str("/v1/pipelines/%s/runs/%s" % (pipeline_name, run_id)), [], {})
            run_details = _decode_dict(json.loads(rez_all_run["response"]))
            

            # do we have an accuracy pipeline for this run?
            accuracy_dataset_name = get_accuracy_pipeline_name(pipeline_name, run_id)
            
            rez_all_run = mldb.perform("GET", str("/v1/pipelines/cls-plugin-%s-%s" % (pipeline_name, run_id)), [], {})
            run_details["config"] = _decode_dict(json.loads(rez_all_run["response"]))

            rez_all_run = mldb.perform("GET", str("/v1/pipelines/cls-plugin-%s-%s/runs/1" % (pipeline_name, run_id)), [], {})
            run_perf = _decode_dict(json.loads(rez_all_run["response"]))
            if "status" in run_perf:
                run_details["eval"] = _decode_dict(run_perf["status"])

            resp_all_run.append(run_details)

        return {"pipeline": pipeline_details,
                "runs": resp_all_run}

    if verb == "PUT" and remaining.startswith("/runeval"):
        payload = json.loads(payload)
        if not "pipeline_name" in payload:
            print payload
            raise Exception("missing key in payload!")

        
        pipeline_name = payload["pipeline_name"]
        run_id = payload["run_id"]
        
        if pipeline_name == "" or run_id == "":
            raise Exception(str("pipeline_name (%s) and run_id (%s) can't be empty!"
                % (pipeline_name, run_id)))
        
        rez = mldb.perform("GET", str("/v1/pipelines/"+pipeline_name), [], {})
        pipeline_conf = json.loads(rez["response"])
        print pipeline_conf["config"]["params"]
        # TODO is 404


        clsBlockName = "clspluginclassifyBlock" + pipeline_name
        print mldb.perform("DELETE", str("/v1/blocks/" + clsBlockName), [], {})
        applyBlockConfig = {
            "id": str(clsBlockName),
            "type": "classifier.apply",
            "params": {
                "classifierUri": str(pipeline_conf["config"]["params"]["classifierUri"])
            }
        }
        print mldb.perform("PUT", str("/v1/blocks/"+clsBlockName), [], applyBlockConfig)

        testClsPipelineName = get_accuracy_pipeline_name(pipeline_name, run_id)
        testClsPipelineConfig = {
            "id": str(testClsPipelineName),
            "type": "accuracy",
            "params": {
                "dataset": { "id": str(payload["testset_id"]) },
                "output": {
                    "id": str("cls-plugin-accuracy-rez-%s-%s" % (pipeline_name, run_id)),
                    "type": "mutable",
                },
                "where": str(payload["where"]),
                "score": str("APPLY BLOCK %s WITH (%s) EXTRACT(score)" % (clsBlockName, pipeline_conf["config"]["params"]["select"])),
                "label": str(payload["label"]),
            }
        }
        print mldb.perform("PUT", str("/v1/pipelines/%s" % testClsPipelineName), [], testClsPipelineConfig)
        print mldb.perform("PUT", str("/v1/pipelines/%s/runs/1" % testClsPipelineName), [], {})

        return "OK!"

    if verb == "GET" and remaining == "/classifier-list":
        rez = mldb.perform("GET", "/v1/pipelines", [], {})
        pipelines = []
        for pipeline in json.loads(rez["response"]):
            # get the piepline details
            rez = mldb.perform("GET", str("/v1/pipelines/%s" % pipeline), [], {})
            resp = json.loads(rez["response"])
            if "type" in resp and resp["type"] != "classifier": continue
            
            # get the runs for the piepeline
            rez_runs = mldb.perform("GET", str("/v1/pipelines/%s/runs" % pipeline), [], {})
            resp_runs = json.loads(rez_runs["response"])

            resp_last_run = {}
            if len(resp_runs) > 0:
                print resp_runs
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

