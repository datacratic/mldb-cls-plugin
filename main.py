
import urllib, datetime, csv


######################
##  Create toy dataset
######################
mldb.perform("DELETE", "/v1/datasets/toy", [], {})

# create a mutable beh dataset
datasetConfig = {
        "type": "beh.mutable",
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
##  Create route to serve routes
######################
mldb.plugin.serve_static_folder("static", "static")
mldb.plugin.serve_documentation_folder('doc')


