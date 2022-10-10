import pandas as pd

from jira_parser import get_soft_skills, get_hard_skills, SkillsSearchType

candidate_ids = pd.read_csv('clients_interviews.csv')
candidate_ids['candidate_id'] = candidate_ids['candidate_id'].astype(int)


staffing_candidates_df = pd.read_csv('Data/staffing_candidates.csv')[['id', 'name_eng']]
staffing_candidates_df = staffing_candidates_df.rename({'id':'candidate_id'}, axis=1)
needed_names_df = candidate_ids.merge(staffing_candidates_df, on='candidate_id', how='left')[['candidate_id', 'name_eng']]

total = 0
got_data = 0
for row in needed_names_df.iloc():
    name = row['name_eng']
    print(get_hard_skills(SkillsSearchType.TECHNOLOGY, name, "2018-01-01", "2022-12-12"))
    got_data += 1
    total += 1
    print(f"{got_data}/{total}")
