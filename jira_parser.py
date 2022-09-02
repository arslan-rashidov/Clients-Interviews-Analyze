import requests

token = "c3B0LmludGVncmF0aW9uOlI4JVV+ZU9+SFo4VA=="

class SkillsSearchType:
    EMPLOYEE = 'employee',
    PROJECT = 'project',
    INTERVIEWER = 'interviewer',
    TECHNOLOGY = 'technology',
    ALL_INTERVIEWERS = 'allInterviewers',
    INTERVIEWER_ACCURACY = 'interviewerAccuracy',
    ACCURACY_PREPARATION = 'AccuracyPreparation'


def get_soft_skills(full_name, start_date, end_date):
    r = requests.get(
        url=f'https://jsupport.andersenlab.com/rest/api/2/search?jql=issuetype="English self-presentation training" AND ("Full name"  ~ "{full_name}" or  text ~ "{full_name}") and created >= {start_date} and created <= {end_date}&fields=summary,status,resolutiondate,created,customfield_13502,issuetype,customfield_12106',
        headers={'Authorization': f'Basic {token}'})

    return r.text


def get_param_by_search_type(search_type, param):
    if search_type == SkillsSearchType.EMPLOYEE:
        return f'cf[14311]~"{param}"'
    elif search_type == SkillsSearchType.INTERVIEWER:
        return f'cf[14311]~"{param}"'
    elif search_type == SkillsSearchType.PROJECT:
        return f'cf[13912]="{param}" or summary~"{param}"'
    elif search_type == SkillsSearchType.TECHNOLOGY:
        return f'cf[15505]="{param}"'
    elif search_type == SkillsSearchType.INTERVIEWER_ACCURACY:
        return f'cf[14311]~"{param}"'
    else:
        return ""


def get_hard_skills(search_type, full_name, start_date, end_date):
    start_date_param = f"and created >= {start_date}"
    end_date_param = f"and created <= {end_date}"

    search_param = f"and {get_param_by_search_type(search_type, full_name)}"

    fields = 'fields=issuetype,customfield_13912,customfield_15505,key,subtasks,summary,assignee,customfield_14200,created,duedate,customfield_14502,customfield_14319,resolution,resolutiondate,status,customfield_15300,customfield_15608,customfield_14310,customfield_14311'

    if search_type == SkillsSearchType.INTERVIEWER:
        url = f"""https://jira.andersenlab.com/rest/api/2/search?jql = project = "Personnel training" and (type = Task and issueFunction in parentsOf("assignee = '{full_name}'") or type = Sub - Task and assignee = '{full_name}') &""" + fields
    elif search_type == SkillsSearchType.INTERVIEWER_ACCURACY:
        url = f"""https://jira.andersenlab.com/rest/api/2/search?jql = project = "Personnel training" and issuetype in (Sub - Task) {search_param} &""" + fields
    else:
        url = f"""https://jira.andersenlab.com/rest/api/2/search?jql = project = "Personnel training" and issuetype in (Task, Sub - Task) {search_param} &""" + fields

    r = requests.get(url=url,
                     headers={'Authorization': f'Basic {token}'})

    return r.text


def get_issues_list(months_ago, max_results=1000):
    url = f"https://jsupport.andersenlab.com/rest/api/2/search?jql=issuetype%20in%20(%22Personnel%20training%22)%20AND%20status%20changed%20to%20%22search%20completed%22%20after%20startOfMonth({-months_ago})%20%20AND%20status%20changed%20to%20%22search%20completed%22%20before%20endOfMonth({-max_results})%20%20AND%20%22Time%20to%20resolution%22%20%3D%20everBreached()%20ORDER%20BY%20updated%20DESC&maxResults={maxResults}"

    r = requests.get(url=url,
                     headers={'Authorization': f'Basic {token}'})

    return r.text

print(get_soft_skills("Armine Saghatelyan", "2021-01-01", "2022-12-12"))
