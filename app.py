from flask import Flask,request,render_template
import numpy as np 
app = Flask(__name__)
@app.route('/')
#Cette fonction retourne la page index.html de notre projet
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST']) #predict sera indiqué dans le form de html comme action pour permettre d'appeler la methode predict
#La fonction ci-dessous fait la prediction
def predict():
    import joblib
    #Nous chargeons notre modele sauvegardé dans le projet Flask pour l'utiliser
    model1=joblib.load('ModelG11.ml')
    model2=joblib.load('ModelG22.ml')
    model3=joblib.load('ModelG33.ml')
    #Nous récupérons toutes les valeurs saisies dans le formulaire html sous forme d'une liste
    string_features=[i for i in request.form.values()]
    string_features1=[i for i in request.form.values()]
    string_features2=[i for i in request.form.values()]
    
    #On recupere toutes les valeurs de la liste sauf la derniere pour G1
    features_model = [string_features[0],string_features[1],string_features[2],string_features[3]
    ,string_features[4],string_features[5],string_features[6],string_features[7],string_features[8]
    ,string_features[9],string_features[10],string_features[11],string_features[12],string_features[13]
    ,string_features[14],string_features[15],string_features[16],string_features[17],string_features[18]
    ,string_features[19],string_features[20],string_features[21],string_features[22],string_features[23]
    ,string_features[24],string_features[25],string_features[26]]

    #On recupere toutes les valeurs de la liste sauf la derniere pour G2
    features_model1 = [string_features1[0],string_features1[1],string_features1[2],string_features1[3]
    ,string_features1[4],string_features1[5],string_features1[6],string_features1[7],string_features1[8]
    ,string_features1[9],string_features1[10],string_features1[11],string_features1[12],string_features1[13]
    ,string_features1[14],string_features1[15],string_features1[16],string_features1[17],string_features1[18]
    ,string_features1[19],string_features1[20],string_features1[21],string_features1[22],string_features1[23]
    ,string_features1[24],string_features1[25],string_features1[26]]

    #On recupere toutes les valeurs de la liste sauf la derniere pour G3
    features_model2 = [string_features2[0],string_features2[1],string_features2[2],string_features2[3]
    ,string_features2[4],string_features2[5],string_features2[6],string_features2[7],string_features2[8]
    ,string_features2[9],string_features2[10],string_features2[11],string_features2[12],string_features2[13]
    ,string_features2[14],string_features2[15],string_features2[16],string_features2[17],string_features2[18]
    ,string_features2[19],string_features2[20],string_features2[21],string_features2[22],string_features2[23]
    ,string_features2[24],string_features2[25],string_features2[26]]
    #On reshape les features pour le rendre un vecteur np capable d'etre introduits dans le modele pour la prediction
    features_model=np.array([features_model]).reshape(1,27)
    features_model1=np.array([features_model1]).reshape(1,27)
    features_mode2=np.array([features_model2]).reshape(1,27) 
    #On predit en utilisant le modele qui a été chargé ci-haut
    prediction1=model1.predict(features_model)[0]
    prediction2=model2.predict(features_model1)[0]
    prediction3=model3.predict(features_model2)[0] 
    #On prepare la chaine de retour contenant la prediction de la date de sortie
    chaine_prediction=" a la probabilité de d'obtenir en G1 "+ (prediction1) + "%, en G2 "+str(prediction2) + "%, et en G3 "+ str(prediction3)    
    #On retourne la page index.html avec le resultat formaté. N.B: prediction_text sera appelé dans la page index.html pour retourner le resultat
    return render_template('index.html',prediction_text='Ce nouveau inscrit {}'.format(chaine_prediction))
#On execute notre application Flask
if __name__ == "__main__":
    app.run(debug=False)

    # if __name__ == "__main__":
    # app.run(debug=False)
    
    
    
    