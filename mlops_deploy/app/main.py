from flask import Flask ,request ,jsonify 
from textblob import TextBlob
import nltk

from googletrans import Translator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle   #dump  #load
from flask_basicauth import BasicAuth
import os 



colunas = ['tamanho','ano','garagem']
lg = pickle.load(open('../../models/modelo.pkl','rb')) #lendo arquivo e atribuindo à variável.

#Definir aplicação
#SERIALIZAMOS!!!!!

# df = pd.read_csv('casas.csv')
# df
# colunas= ['tamanho','ano','garagem']  #parsing dos dados 

# X= df.drop(['preco'],axis=1)
# y= df['preco']


# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.3)

# lg = LinearRegression()
# lg.fit(X_train,y_train)



app = Flask('__name__')

app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)
#Definir rotas da API (endpoints)

@app.route('/')
def home():
    return "Minha primeira API"


@app.route('/sentimento/<frase>') #receber info do usuário: Na própria URL
# @basic_auth.required
def sentimento(frase): #recebe frase vindo da URL
    
    tr= Translator()
    
    frase_en = tr.translate(frase,dest='en')
    
    tb_en = TextBlob(frase_en.text)
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST']) #Tirar da URL , e colocar POST --> Consigo receber PAYLOAD.  Não dá pra fazer na URL, agora é no postman  
@basic_auth.required
def cotacao():
    dados = request.get_json() #pega json que o usuário mandar - PAYLOAD
    dados_input = [ dados[col]  for col in colunas]
    preco_prev=lg.predict([dados_input])
    return jsonify(preco_prev[0])  #string retorno de API


app.run(debug=True,host='0.0.0.0') #debug=True, restarta a execução no terminal. identifica alterações.   #host 0.0.0.0 para todos os ambientes


#CUIDADO COM VERSÕES!
import sklearn
sklearn.__version__