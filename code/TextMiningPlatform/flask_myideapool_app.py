############### APP ###########################################################
from flask import Flask, jsonify, abort, make_response, request, url_for

app = Flask(__name__)
app.debug = True
app.secret_key = 'super-secret'  # Change this!

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

################# Flask JWT #########################################################
from werkzeug.security import safe_str_cmp

from flask_jwt_extended import JWTManager, jwt_required, \
    create_access_token, jwt_refresh_token_required, \
    create_refresh_token, get_jwt_identity

# Setup the Flask-JWT-Extended extension
jwt = JWTManager(app)

id_counter = 0
class User(object):
    def __init__(self, email, name, password):
        global id_counter
        self.id = id_counter + 1
        self.email = email
        self.name = name
        self.password = password

    def __str__(self):
        return "User(id={})".format(str(self.id))

users = [
    User('name1@email1.com', 'name1','abcXYZ123'),
    User('name2@email2.com', 'name2', 'xyzABC123'),
]

useremail_table = {u.email: u for u in users}
userid_table = {u.id: u for u in users}

# Atleast 8 characters, including 1 uppercase letter, 1 lowercase letter, and 1 number
def check_password(passwd_str):
    letters = set(passwd_str)
    mixed = any(letter.islower() for letter in letters) and any(letter.isupper() for letter in letters) and any(letter.isdigit() for letter in letters)
    if len(passwd_str) > 8 and mixed:
        return passwd_str
    return ""

def authenticate(email, password):
    user = useremail_table.get(email)
    if user and safe_str_cmp(user.password.encode('utf-8'), password.encode('utf-8')):
        return user

def identity(payload):
    user_id = payload['identity']
    return userid_table.get(user_id, None)

@app.route('/access-tokens', methods=['POST'])
def access_tokens_login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    params = request.get_json()
    email = params.get('email', None)
    password = params.get('password', None)
    # print("name {}".format(params))

    if not email:
        return jsonify({"msg": "Missing email parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    user = authenticate(email,password)
    if not user:
        return jsonify({"msg": "Bad email or password"}), 401

    access_token = create_access_token(identity=email)
    refresh_token = create_refresh_token(identity=email)
    ret = {
        'jwt': access_token,
        'refresh_token': refresh_token
    }
    return jsonify(ret), 200

@app.route('/users', methods=['POST'])
def sign_up():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    params = request.get_json()
    email = params.get('email', None)
    name = params.get('name', None)
    password = params.get('password', None)

    if not email:
        return jsonify({"msg": "Missing email parameter"}), 400
    if not name:
        return jsonify({"msg": "Missing name parameter"}), 400
    if not password and check_password(password) == "":
        return jsonify({"msg": "Missing correct password parameter"}), 400

    new_user = User(email,name,password)
    users.append(new_user)
    useremail_table[email] = new_user
    userid_table[new_user.id] = new_user

    ret = {
        'email': email,
        'name': name,
        'password': password
    }
    return jsonify(ret), 200

@app.route('/access-tokens', methods=['DELETE'])
@jwt_required
def access_tokens_logout():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    params = request.get_json()
    refresh_token = params.get('refresh_token', None)
    if not refresh_token:
        return jsonify({"msg": "Missing refresh_token parameter"}), 400

    ret = {
        'refresh_token': create_refresh_token(identity=get_jwt_identity())
    }

    return jsonify(ret), 200

@app.route('/me', methods=['GET'])
@jwt_required
def me():
    return jsonify(get_jwt_identity()), 200

@app.route('/access-tokens/refresh', methods=['POST'])
@jwt_required
def refresh():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    params = request.get_json()
    # print("Inside /access-tokens/refresh. params {}".format(params))
    refresh_token = params.get('refresh_token', None)
    if not refresh_token:
        return jsonify({"msg": "Missing refresh_token parameter"}), 400
    # print("Inside /access-tokens/refresh. refresh token {}".format(refresh_token))

    email = get_jwt_identity()
    # print("Inside /access-tokens/refresh. current user (email) {}".format(email))

    current_user = useremail_table.get(email, None)
    if not authenticate(current_user.email,current_user.password):
        return jsonify({"msg": "Bad refresh token"}), 401

    ret = {
        'refresh_token': create_refresh_token(identity=current_user.email)
    }

    return jsonify(ret), 200

# # Get token first
# curl -H "Authorization: Bearer $ACCESS" http://localhost:5000/protected
# curl -H "x-access-token:$ACCESS" http://localhost:5000/protected
@app.route('/protected', methods=['GET'])
@jwt_required
def protected():
    # Access the identity of the current user with get_jwt_identity
    # current_user = get_jwt_identity()
    # return jsonify({'hello_from': current_user}), 200
    user = get_jwt_identity()
    return jsonify({'email is': user}), 200

############### Programm ###########################################################


ideas = [
    {
        'id': 1,
        'content': u'Buy groceries',
        'impact': 1,
        'ease': 1,
        'confidence ': 1,
        'average_score': 1,
        'created_at': 1506940089
    },
    {
        'id': 2,
        'content': u'Buy Something',
        'impact': 1,
        'ease': 1,
        'confidence ': 1,
        'average_score': 1,
        'created_at': 1506940089
    }
]

def make_public_idea(idea):
    new_task = {}
    for field in idea:
        if field == 'id':
            new_task['uri'] = url_for('get_idea', idea_id=idea['id'], _external=True)
        else:
            new_task[field] = idea[field]
    return new_task

@app.route('/ideas', methods=['POST'])
@jwt_required
def create_idea():
    # print("Inside create idea with raw request : {}".format(request.json))
    if not request.json or not 'content' in request.json:
        abort(400)

    new_id = ideas[-1]['id'] + 1
    new_content = request.json['content']
    new_impact = int(request.json.get('impact', ""))
    new_ease = int(request.json.get('ease', ""))
    new_confidence = int(request.json.get('confidence', ""))
    new_total_score = new_impact + new_confidence + new_ease
    new_avergae_score = new_total_score/3.0
    new_created_at = "1506940089" # <TODO> current time stamp??

    new_idea = {
        'id': new_id,
        'content': new_content,
        'impact': new_impact,
        'ease': new_ease,
        'confidence': new_confidence,
        'average_score': new_avergae_score,
        'created_at': new_created_at
    }
    ideas.append(new_idea)
    # print("Ideas so far : {}".format(ideas))
    return jsonify({'idea': new_idea}), 200

@app.route('/ideas/<int:idea_id>', methods=['DELETE'])
@jwt_required
def delete_idea(idea_id):
    # print("Inside delete idea with id : {}".format(idea_id))
    idea = [idea for idea in ideas if idea['id'] == idea_id]
    if len(idea) == 0:
        abort(404)
    ideas.remove(idea[0])
    return jsonify({'result': True})


@app.route('/ideas', methods=['GET'])
@jwt_required
def get_ideas():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    params = request.get_json()
    page = int(params.get('page', None))
    if not page:
        return jsonify({"msg": "Missing page number parameter"}), 400
    # <TODO> build page number logic
    return jsonify({'ideas': ideas})

@app.route('/ideas/<int:idea_id>', methods=['PUT'])
@jwt_required
def update_idea(idea_id):
    idea = [idea for idea in ideas if idea['id'] == idea_id]
    if len(idea) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if not 'content' in request.json:
        abort(400)
    if not 'impact' in request.json:
        abort(400)
    if not 'ease' in request.json:
        abort(400)
    if not 'confidence' in request.json:
        abort(400)
    # print("Inside Update idea with raw request data as : {}".format(request.json))

    updated_content = request.json.get('content', idea[0]['content'])
    updated_impact = int(request.json.get('impact', idea[0]['impact']))
    updated_ease = int(request.json.get('ease', ""))
    updated_confidence = int(request.json.get('confidence', ""))
    updated_total_score = updated_impact + updated_confidence + updated_ease
    updated_avergae_score = updated_total_score/3.0
    updated_created_at = "1506940089" # <TODO> current time stamp??

    idea[0]['content'] = updated_content
    idea[0]['impact'] = updated_impact
    idea[0]['ease'] = updated_ease
    idea[0]['confidence'] = updated_confidence
    idea[0]['average_score'] = updated_avergae_score
    idea[0]['created_at'] = updated_created_at
    return jsonify({'idea': idea[0]})



if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=False)


