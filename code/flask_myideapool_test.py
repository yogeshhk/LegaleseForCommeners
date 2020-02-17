import json
import requests


def run_tests(base_url):
    print("Testing url: {} ...".format(base_url))

    # Sign up: Without JWT
    # curl -H "Content-Type: application/json" -X POST -d "{"""email""":"""email1""","""name""":"""name1""","""password""":"""abcxyz"""}" http://localhost:5000/access-tokens
    headers = {"Content-Type": "application/json"}
    endpoint = base_url + "/users"
    dicttosend = {"email":"name3@email3.com","name":"name3","password":"Abcxyz123"}
    response = requests.post(endpoint,json=dicttosend,headers=headers).json()
    print("New User signed up : {}".format(response))

    # User login: Get Access token given login details
    # curl -H "Content-Type: application/json" -X POST -d "{"""email""":"""email1""","""password""":"""abcxyz"""}" http://localhost:5000/access-tokens
    headers = {"Content-Type": "application/json"}
    endpoint = base_url + "/access-tokens"
    dicttosend = {"email":"name1@email1.com","password":"abcXYZ123"}
    response = requests.post(endpoint,json=dicttosend,headers=headers).json()
    access_token = response.get("jwt",None)
    print("User found, current, Access Token : {}".format(access_token))
    refresh_token = response.get("refresh_token",None)
    print("User found, current, Refresh Token : {}".format(refresh_token))



    # Refresh JWT
    # curl -H "x-access-token:$ACCESS,Content-Type: application/json"  -X POST -d "{"""refresh_token""":$REFRESH}" http://localhost:5000/access-tokens/refresh
    headers = {"Authorization": "Bearer {}".format(access_token), "Content-Type": "application/json"}
    # headers = {"x-access-token" : "{}".format(access_token), "Content-Type": "application/json"}
    endpoint = base_url + "/access-tokens/refresh"
    dicttosend = {"refresh_token":refresh_token}
    response = requests.post(endpoint,json=dicttosend,headers=headers).json()
    refresh_token = response.get("refresh_token",None)
    print("Refreshed JWT, New Refresh Token : {}".format(refresh_token))

    # User logout
    # curl -H "x-access-token:$ACCESS,Content-Type: application/json"   -X DELETE -d "{"""refresh_token""":$REFRESH}" http://localhost:5000/access-tokens
    headers = {"Authorization": "Bearer {}".format(access_token), "Content-Type": "application/json"}
    # headers = {"x-access-token" : "{}".format(access_token), "Content-Type": "application/json"}
    endpoint = base_url + "/access-tokens"
    dicttosend = {"refresh_token":refresh_token}
    response = requests.delete(endpoint,json=dicttosend,headers=headers).json()
    refresh_token = response.get("refresh_token",None)
    print("User deleted, New Refresh Token : {}".format(refresh_token))

    # Get current user's info
    # curl -H "Authorization": "Bearer $ACCESS,Content-Type: application/json" -X GET http://localhost:5000/me
    headers = {"Authorization": "Bearer {}".format(access_token)}
    # headers = {"x-access-token" : "{}".format(access_token)}
    endpoint = base_url + "/me"
    response = requests.get(endpoint,headers=headers).json()
    print("Current User (email): {}".format(response))

    # Create Idea
    headers = {"Authorization": "Bearer {}".format(access_token), "Content-Type": "application/json"}
    # headers = {"x-access-token" : "{}".format(access_token), "Content-Type": "application/json"}
    endpoint = base_url + "/ideas"
    dicttosend = {"content":"my first idea", "impact":"5", "ease":"4","confidence":"4"}
    response = requests.post(endpoint,json=dicttosend,headers=headers).json()
    new_idea = response.get("idea",None)
    print("Created new idea as : {}".format(new_idea))


    # Delete Idea
    headers = {"Authorization": "Bearer {}".format(access_token), "Content-Type": "application/json"}
    # headers = {"x-access-token" : "{}".format(access_token), "Content-Type": "application/json"}
    endpoint = base_url + "/ideas/2"
    response = requests.delete(endpoint,headers=headers).json()
    print("Idea deletion had response : {}".format(response))

    # Get Ideas (by giving page id)
    headers = {"Authorization": "Bearer {}".format(access_token), "Content-Type": "application/json"}
    # headers = {"x-access-token" : "{}".format(access_token), "Content-Type": "application/json"}
    endpoint = base_url + "/ideas"
    dicttosend = {"page":"1"}
    response = requests.get(endpoint,json=dicttosend,headers=headers).json()
    ideas_on_page = response.get("ideas",None)
    print("Get ideas on page : {}".format(ideas_on_page))

    # Update Idea
    headers = {"Authorization": "Bearer {}".format(access_token), "Content-Type": "application/json"}
    # headers = {"x-access-token" : "{}".format(access_token), "Content-Type": "application/json"}
    endpoint = base_url + "/ideas/1"
    dicttosend = {"content":"my updated idea", "impact":"2", "ease":"2","confidence":"2"}
    response = requests.put(endpoint,json=dicttosend,headers=headers).json()
    new_idea = response.get("idea",None)
    print("Updated new idea as : {}".format(new_idea))

if __name__ == '__main__':
    local_server_location = "http://localhost:5000"
    remote_server_location = "https://myideapool.herokuapp.com"

    run_tests(local_server_location)
    run_tests(remote_server_location)




