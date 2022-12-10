from flask import Flask, jsonify,request

app=Flask(__name__)



@app.route("/uImg", methods=['GET','POST'])
def val_img():
    if request.method=="POST":
        d=request.get_data()
        print(d)
        return jsonify({"img":"post is working"})
    elif request.method=="GET":
        #data=request.args.get("img")
        d=request.args.get("img")
        return jsonify({"img":"get is working"})
app.run(debug=True)