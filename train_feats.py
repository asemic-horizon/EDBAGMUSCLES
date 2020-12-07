import aiosql, sqlite3
import numpy as np

spacer = lambda i: "\n" if i%5  == 0 else ""
feat_cols = ",".join([f"cast(feature_{i} as float){spacer(i)}" for i in range(129)])
resp_cols = ",".join([f"cast({col} as float)" for col in ["weight","resp"]])
sql_query = f"""
-- name: random_feat_obs

select {resp_cols},{feat_cols} from train where weight != '0' and random() %2 limit :num;
"""

queries = aiosql.from_str(sql_query,"sqlite3")
conn = sqlite3.connect("train_data.db")

def get_data(num):
	return np.array(queries.random_feat_obs(num=num,conn=conn))
