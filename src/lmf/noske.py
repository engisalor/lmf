from sgex.job import Job

concept = "forced displacement"  # used as a reference by the llm
corpus = "rw_en24"
query = 'alemma,"forced""displacement"'
random_sample = 500
params = {
    "call_type": "View",
    "corpname": corpus,
    "q": [query, f"r{random_sample}"],
    "viewmode": "sen",
    "pagesize": 500,  # sample size (redundant if random_sample)
}
j = Job(params=params, verbose=True)
j.run()
## convert concordances to prompt input
lines = [
    {"token_number": x["toknum"], "sentence": " ".join(x["token"].values())}
    for x in j.data.view[0].lines_to_tacred()
]
inputs = [{"input": x["sentence"]} for x in lines]

# TODO save to yaml
