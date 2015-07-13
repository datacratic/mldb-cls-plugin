
######################
##  Handle custom routes
######################
print "Handling route in python"
import json, re, datetime

rp = mldb.plugin.rest_params

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

def get_accuracy_procedure_name(procedure, run):
    return "cls-plugin-%s-RUN:%s" % (procedure, run)


if rp.verb == "GET" and rp.remaining == "/dataset-details":
    datasets = []
    rez = mldb.perform("GET", "/v1/datasets", [], {})
    for dataset in json.loads(rez["response"]):
        rez = mldb.perform("GET", str("/v1/datasets/%s" % dataset), [], {})
        resp = json.loads(rez["response"])

        # skip the datasets we generate using the cls plugin
        if dataset.startswith("cls-plugin-"): continue

        datasets.append({
            "id": str(dataset),
            "type": str(resp["type"])
        })

    mldb.plugin.set_return(datasets)

if rp.verb == "GET" and rp.remaining.startswith("/cls-details"):
    procedure_name = rp.remaining.split("/")[-1]
    rez = mldb.perform("GET", "/v1/procedures/"+procedure_name, [], {})
    if rez["statusCode"] == 404:
        raise Exception("Procedure does not exist!")
    prez = json.loads(rez["response"])
    procedure_details = _decode_dict(prez)

    rez_runs = mldb.perform("GET", str("/v1/procedures/%s/runs" % procedure_name), [], {})
    resp_runs = json.loads(rez_runs["response"])

    resp_all_run = []
    for run_id in resp_runs:
        rez_all_run = mldb.perform("GET", str("/v1/procedures/%s/runs/%s" % (procedure_name, run_id)), [], {})
        run_details = _decode_dict(json.loads(rez_all_run["response"]))
        run_details["ran_eval"] = False

        # do we have an accuracy procedure for this run?
        procedureRunName = get_accuracy_procedure_name(procedure_name, run_id)

        rez_all_run = mldb.perform("GET", str("/v1/procedures"), [], {})
        for pname in json.loads(rez_all_run["response"]):
            if pname.startswith(procedureRunName):
                eval_details = {}
                run_details["ran_eval"] = True

                rez_all_run = mldb.perform("GET", str("/v1/procedures/%s" % pname), [], {})
                eval_details["config"] = _decode_dict(json.loads(rez_all_run["response"]))

                eval_procedure_runs = mldb.perform("GET", str("/v1/procedures/%s/runs" % pname), [], {})
                eval_procedure_runs = json.loads(eval_procedure_runs["response"])
                if "error" not in eval_procedure_runs and len(eval_procedure_runs) > 0:
                    run_id = eval_procedure_runs[-1]
                    rez_all_run = mldb.perform("GET", 
                            str("/v1/procedures/%s/runs/%s" % (pname, run_id)), [], {})
                    run_perf = _decode_dict(json.loads(rez_all_run["response"]))
                    if "state" in run_perf:
                        eval_details["eval"] = _decode_dict(run_perf)

                run_details["eval"] = eval_details
                break

        resp_all_run.append(run_details)

    mldb.plugin.set_return({"procedure": procedure_details,
                            "runs": resp_all_run})

if rp.verb == "PUT" and rp.remaining.startswith("/runeval"):
    payload = json.loads(rp.payload)
    if not "procedure_name" in payload:
        print payload
        raise Exception("missing key in payload!")

    procedure_name = payload["procedure_name"]
    
    if procedure_name == "":
        raise Exception(str("procedure_name (%s) can't be empty!"
            % (procedure_name)))
    
    rez = mldb.perform("GET", str("/v1/procedures/"+procedure_name), [], {})
    if rez["statusCode"] == 404:
        raise Exception("Procedure does not exist!")

    procedure_conf = json.loads(rez["response"])

    # get latest procedure run
    rez = mldb.perform("GET", str("/v1/procedures/%s/runs" % procedure_name), [], {})
    procedure_runs = json.loads(rez["response"])
    lastRun = procedure_runs[-1]
    procedureRunName = get_accuracy_procedure_name(procedure_name, lastRun)


    clsFunctionName = "%s-classifyFunction-" % procedureRunName
    print mldb.perform("DELETE", str("/v1/functions/" + clsFunctionName), [], {})
    applyFunctionConfig = {
        "id": str(clsFunctionName),
        "type": "classifier",
        "params": {
            "modelFileUrl": str(procedure_conf["config"]["params"]["modelFileUrl"])
        }
    }
    print mldb.perform("PUT", str("/v1/functions/"+clsFunctionName), [], applyFunctionConfig)

    now = datetime.datetime.now().isoformat()
    testClsPipeName = procedureRunName + "-" + now
    testClsProcedureConfig = {
        "id": str(testClsPipeName),
        "type": "classifier.test",
        "params": {
            "dataset": { "id": str(payload["testset_id"]) },
            "output": {
                "id": str("%s-rez" % testClsPipeName),
                "type": "beh.mutable",
            },
            "where": str(payload["where"]),
            "score": str("APPLY FUNCTION \"%s\" WITH (object(SELECT %s) AS features) EXTRACT(score)" %
                (clsFunctionName, procedure_conf["config"]["params"]["select"])),
            "label": str(payload["label"]),
        }
    }
    print mldb.perform("PUT",  str("/v1/procedures/%s" % testClsPipeName), [], testClsProcedureConfig)
    print mldb.perform("POST", str("/v1/procedures/%s/runs" % testClsPipeName), [], {})

    mldb.plugin.set_return("OK!")

if rp.verb == "GET" and rp.remaining.startswith("/roccurve"):
    dataset_name = rp.remaining.split("/")[-1]
    rez = mldb.perform("GET", "/v1/datasets/"+dataset_name+"/query", [], {})
    if rez["statusCode"] == 404:
        raise Exception("Procedure does not exist!")
    prez = json.loads(rez["response"])
    curve = _decode_list(prez)

    new_curve = []
    for pt in curve:
        new_pt = {}
        for col in pt["columns"]:
            #if col[0] not in ["falsePositiveRate", "truePositiveRate", "index"]: continue
            new_pt[col[0]] = col[1]
        new_curve.append(new_pt)
    mldb.plugin.set_return(new_curve)


if rp.verb == "GET" and rp.remaining == "/classifier-list":
    rez = mldb.perform("GET", "/v1/procedures", [], {})
    mldb.log(str(rez))
    procedures = []
    for procedure in json.loads(rez["response"]):
        # get the piepline details
        rez = mldb.perform("GET", str("/v1/procedures/%s" % procedure), [], {})
        mldb.log(str(rez))
        resp = json.loads(rez["response"])
        if "type" in resp and resp["type"] != "classifier.train":
            continue
        #if resp["id"].startswith("cls-plugin-"): continue

        # get the runs for the piepeline
        rez_runs = mldb.perform("GET", str("/v1/procedures/%s/runs" % procedure), [], {})
        resp_runs = json.loads(rez_runs["response"])

        resp_last_run = {}
        if rez_runs["statusCode"] != 404 and len(resp_runs) > 0:
            try:
                rez_last_run = mldb.perform("GET", str("/v1/procedures/%s/runs/%s" % (procedure, resp_runs[-1])), [], {})
                resp_last_run = json.loads(rez_last_run["response"])
                runs = _decode_list(resp_runs)
            except Exception, e:
                mldb.log(str(e))
                runs = []

        else:
            runs = []

        procedures.append({
            "id": str(procedure),
            "state": str(resp["state"]),
            "type": str(resp["type"]) if "type" in resp else "", #check no longer required when MLDB-572 is fixed
            "params": _decode_dict(resp["config"]["params"]) if "config" in resp else {},
            "runs": runs,
            "last_run": _decode_dict(resp_last_run)
        })
    mldb.plugin.set_return(procedures)

if rp.verb == "GET" and rp.remaining == "/cls-presets":
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
    mldb.plugin.set_return(configs)

if rp.verb == "DELETE" and rp.remaining.startswith("/del-procedure/"):
    remaining_split = rp.remaining.split("/")
    if len(remaining_split) != 3:
        raise Exception("Need to specify procedure name!")
    procedure_name = remaining_split[-1]
    rez = mldb.perform("GET", "/v1/procedures/"+procedure_name, [], {})
    if rez["statusCode"] == 404:
        raise Exception("Procedure does not exist!")

    deleteLog = []
    def doDelete(route):
        rez = mldb.perform("DELETE", route, [], {})
        msg = "Deleting %s. Status: %d" % (route, rez["statusCode"])
        print msg
        deleteLog.append(msg)

    # delete all derived entities
    for entity_type in ["functions", "datasets", "procedures"]:
        rez = mldb.perform("GET", str("/v1/"+entity_type), [], {})
        for entity in json.loads(rez["response"]):
            if entity.startswith("cls-plugin-%s-RUN" % procedure_name):
                full_entity = str("/v1/%s/%s" % (entity_type, entity))
                doDelete(full_entity)

    # delete the main procedure
    doDelete(str("/v1/procedures/" + procedure_name))

    mldb.plugin.set_return(deleteLog)

if rp.verb == "POST" and rp.remaining == "/loadcsv":
    payload = json.loads(rp.payload)
    reader = csv.DictReader(open(urllib.urlretrieve(payload["url"])[0]))
    dataset = mldb.create_dataset(dict(id=str(payload["name"]), type="beh.mutable"))
    for i, row in enumerate(reader):
        values = []
        row_name = i
        for col in row:
            if col == "":
                row_name = row[col]
            else:
                values.append([col, row[col], 0])
        dataset.record_row(row_name, values)
    dataset.commit()
    mldb.plugin.set_return("yay")

else:
    mldb.log(rp.verb)
    mldb.log(rp.remaining)


