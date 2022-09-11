from xml.etree.ElementTree import tostring
from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
#from ipynb.fs.full.my_functions import give_recomen

app = Flask(__name__)

user_rtng_pvt = pickle.load(open("user_rtng_pvt.pkl", "rb"))
modelknn = pickle.load(open("knn.pkl", "rb"))
model_indices = pickle.load(open("indices_cbr.pkl", "rb"))
model_indices_vendor_srtd= pickle.load(open("indices_vendor_sorted.pkl", "rb"))
sigmoid = pickle.load(open("sigmoid.pkl", "rb"))
vendors = pickle.load(open("vendors.pkl", "rb"))
rating_vendor = pickle.load(open("rating_vendor.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

def get_ip():
    qi = request.form.get("fname")
    # query_index=int(query_index)
    query_index=model_indices_vendor_srtd[qi]
    print(query_index)
    return query_index

def get_ip2():
    qi = request.form.get("fname")
    # query_index=int(query_index)
    query_index2=model_indices[qi]
    print(query_index2)
    return query_index2

@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # query_index = 20
        query_index = get_ip()

        distances, indices = modelknn.kneighbors(user_rtng_pvt.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
        user_rtng_pvt.iloc[query_index,:].values.reshape(1,-1)

        str_new=[]
        str2_wa=[]
        for i in range(0, len(distances.flatten())):
            if i == 0:
                str=user_rtng_pvt.index[query_index]
                print("YOU HAVE SELECTED - ",str)
                # print(model)
            else:
                str1 = user_rtng_pvt.index[indices.flatten()[i]]
                print(i,".", str1)
                str2=rating_vendor['weighted_average'].iloc[model_indices[str1]]
                str2_wa.append(str2)
                str_new.append(str1)

        output = dict() 
        for index,value in enumerate(str_new):
            output[index+1] = value
        print(output)

        # Content based Recommendation

        def give_rec(title, sig=sigmoid):
            # Get the index corresponding to vendor
            print(title)
            idx = model_indices_vendor_srtd[title]
            #idx = model_indices1['GREENPOINT HEIGHTS']
            print(idx)
            # Get the pairwsie similarity scores 
            sig_scores = list(enumerate(sig[idx]))

            # Sort the vendors 
            sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

            # Scores of the 5 most similar movies
            sig_scores = sig_scores[1:6]

            # Vendor indices
            vendor_indices = [i[0] for i in sig_scores]

            # Top 5 most similar vendors
            return vendors['vendor_name'].iloc[vendor_indices]

        query_index2=get_ip2()
        op= give_rec(query_index2)

        print(op)
        op1 = op.values.tolist()
        op2=", ".join(op1)

        str_cb=[]
        str_cb.append(op2)

        newlist = []
        for word in str_cb:
            word = word.split(", ")
            newlist.extend(word)  # <----
        

        str5_wa= rating_vendor['weighted_average'].iloc[model_indices[newlist[0]]]
        str6_wa= rating_vendor['weighted_average'].iloc[model_indices[newlist[1]]]
        str7_wa= rating_vendor['weighted_average'].iloc[model_indices[newlist[2]]]
        str8_wa= rating_vendor['weighted_average'].iloc[model_indices[newlist[3]]]
        str9_wa= rating_vendor['weighted_average'].iloc[model_indices[newlist[4]]]
    
        return render_template('predict.html',prediction_text="{}".format(output),restaurant_name=str,content_based=op2,
        rec_1=str_new[0],
        rec_2=str_new[1],
        rec_3=str_new[2],
        rec_4=str_new[3],
        rec_5=str_new[4],

        rec_6=newlist[0],
        rec_7=newlist[1],
        rec_8=newlist[2],
        rec_9=newlist[3],
        rec_10=newlist[4],

        wa_1=str2_wa[0],
        wa_2=str2_wa[1],
        wa_3=str2_wa[2],
        wa_4=str2_wa[3],
        wa_5=str2_wa[4],
        wa_6=str5_wa,
        wa_7=str6_wa,
        wa_8=str7_wa,
        wa_9=str8_wa,
        wa_10=str9_wa,
        )


    return render_template("home.html")



if __name__ == "__main__":
    app.run(debug=True)