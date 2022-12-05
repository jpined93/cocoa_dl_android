from flask import Flask, jsonify,request

app=Flask(__name__)



@app.route("/uImg", methods=['GET','POST'])
def val_img():
    if request.method=="POST":
        data=request.args
        print(data)
        return jsonify({"img":"post is working"})
    elif request.method=="GET":
        data=request.args.get("img")
        print(data)
        return jsonify({"img":"get is working"})

app.run(debug=True)