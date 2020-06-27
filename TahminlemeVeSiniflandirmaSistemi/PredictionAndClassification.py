import numpy as n
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import  train_test_split as tts
import pandas as p
from sklearn.linear_model import LinearRegression as lineerreg
from sklearn.linear_model import LogisticRegression as LG
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
import statsmodels.regression.linear_model as sfa 
from sklearn.metrics import r2_score as rkare

#----------- PREPROCESSİNG AŞAMASI ---------------
#Dosya okundu
readCSV = p.read_csv("datasetler/SoccerGame.csv")
df = p.DataFrame(readCSV)

#isim ve soyisim bilgileri yok edildi
df.drop(["first_name","last_name"],axis=1, inplace=True)
x = df["position"].unique()

#maaş ve tazminattaki nan değerler kaldırıldı
I = Imputer(missing_values="NaN", strategy="mean",axis=0)
processedValues = I.fit_transform(df.iloc[:,[2,3]])
df.iloc[:,[2,3]]= processedValues

#club ve positiondaki nan değerler kaldırıldı
df.dropna(inplace=True)

#One Hot Encoder kullanılarak club ve position bilgisi binary sayılara dönüştürülecek
#club bilgisi binary hale getirildi

ohe = OneHotEncoder(categorical_features="all")
clublar = ohe.fit_transform(df.iloc[:,[0]]).toarray()
clubisimleri = df["club"].unique()
clubisimleri.sort()

clubDataframe = p.DataFrame(data = clublar, columns = clubisimleri)

#position bilgisi binary hale getirildi
ohe2 = OneHotEncoder(categorical_features="all")
pozisyonlar = ohe.fit_transform(df.iloc[:,[1]]).toarray()
pozisyonisimleri = df["position"].unique()
pozisyonisimleri.sort()

pozisyonDataframe = p.DataFrame(data = pozisyonlar, columns = pozisyonisimleri)

#Oluşturulan club ve position binary bilgileri birleştirildi
df.drop(["club","position"],axis=1,inplace=True)
#yeniveri = p.concat([clubDataframe, pozisyonDataframe, df ], axis=1)
#print(yeniveri)

#LİNEER REGRESYON KULLANILARAK TAKIM VE POZİSYON BİLGİSİNE GÖRE ALDIKLARI MAAŞ VE TAZMİNATI TAHMİN EDECEĞİZ

#train ve test club için
clubEgitim, clubtest, maasvetazminategitim,maasvetazminattest = tts(clubDataframe,df, test_size=0.33)
clubEgitim= clubEgitim.sort_index()
clubtest= clubtest.sort_index()
maasvetazminategitim= maasvetazminategitim.sort_index()
maasvetazminattest= maasvetazminattest.sort_index()

#club için lineer regresoyon bağıntısını hesaplama
lineer= lineerreg()
lineer.fit(clubEgitim,maasvetazminategitim)
linetahC=lineer.predict(clubtest)
linetahC = p.DataFrame(linetahC)

#CLUBA GÖRE MAAŞ VE TAZMİNAT TAHMİNLERİNİ GÖRSELLEŞTİRME
maasvetazminattest = p.DataFrame(maasvetazminattest.values)

###############
pozisyonEgitim1, pozisyontest1, maasvetazminategitim1,maasvetazminattest1 = tts(pozisyonDataframe ,df, test_size=0.33)
pozisyonEgitim1= pozisyonEgitim1.sort_index()
pozisyontest1= pozisyontest1.sort_index()
maasvetazminategitim1= maasvetazminategitim1.sort_index()
maasvetazminattest1= maasvetazminattest1.sort_index()

#pozisyon için lineer regresyon bağıntısı hesaplama
lineer2 = lineerreg()
lineer2.fit(pozisyonEgitim1,maasvetazminategitim1)
linetahP=lineer2.predict(pozisyontest1)
linetahP = p.DataFrame(linetahP)

#POZİSYONA GÖRE MAAŞ VE TAZMİNAT TAHMİNLERİNİ GÖRSELLEŞTİRME    
maasvetazminattest1 = p.DataFrame(maasvetazminattest1.values)
###############
maasegitim,maastest,tazminategitim,tazminattest= tts(df.iloc[:,[0]],df.iloc[:,[1]], test_size=0.33)
maasegitim = maasegitim.sort_index()
maastest = maastest.sort_index()
tazminategitim = tazminategitim.sort_index()
tazminattest = tazminattest.sort_index()

lineer3= lineerreg()
lineer.fit(maasegitim,tazminategitim)
tahminn = lineer.predict(maastest)
################

class Prediction():
    def clubGorsellestir(self):   
   
        plt.title("CLUBA GÖRE LİNEER REGRESYON GRAFİĞİ")
        plt.plot(maasvetazminattest, color="blue", label="Gerçek Değerler")
        plt.plot(linetahC, color="red", lw=3, label= "Tahmini Değerler")
        plt.legend()
        plt.show()

        plt.title("CLUBA GÖRE GERÇEK DEĞERLER ")
        plt.pcolormesh(maasvetazminattest)
        plt.show()
        
        plt.title("CLUBA GÖRE LİNEER REGRESYON TAHMİNİ DEĞERLERİ ")
        plt.pcolormesh(linetahC)  
        plt.show()

    def clubTahmin(self):
    
        clubEgitim, clubtest, maasvetazminategitim,maasvetazminattest = tts(clubDataframe,df, test_size=0.33)
        clubEgitim= clubEgitim.sort_index()
        clubtest= clubtest.sort_index()
        maasvetazminategitim= maasvetazminategitim.sort_index()
        maasvetazminattest= maasvetazminattest.sort_index()

        #club için lineer regresoyon bağıntısını hesaplama
        lineer= lineerreg()
        lineer.fit(clubEgitim,maasvetazminategitim)
    
        index=0
        tahmindf=p.DataFrame(columns=clubisimleri)
        for takim in clubisimleri:
            print(index, takim)
            index+=1
        takim=input("Maaşını tahmin etmek istediğiniz futbolcunun takımını giriniz: ")
        takim = takim.upper() 
        tahmindf.loc[0]=0
        tahmindf[takim]=1
        tahminsonuc = lineer.predict(tahmindf)
        tahminsonuc= p.DataFrame(tahminsonuc, columns=["Tahmini Maaş","Tahmini Tazminat"]).iloc[0]
        print(tahminsonuc)   

    #train ve test pozisyon için
    pozisyonEgitim1, pozisyontest1, maasvetazminategitim1,maasvetazminattest1 = tts(pozisyonDataframe ,df, test_size=0.33)
    pozisyonEgitim1= pozisyonEgitim1.sort_index()
    pozisyontest1= pozisyontest1.sort_index()
    maasvetazminategitim1= maasvetazminategitim1.sort_index()
    maasvetazminattest1= maasvetazminattest1.sort_index()

    #pozisyon için lineer regresyon bağıntısı hesaplama
    lineer2 = lineerreg()
    lineer2.fit(pozisyonEgitim1,maasvetazminategitim1)
    linetahP=lineer2.predict(pozisyontest1)
    linetahP = p.DataFrame(linetahP)

    #POZİSYONA GÖRE MAAŞ VE TAZMİNAT TAHMİNLERİNİ GÖRSELLEŞTİRME    
    maasvetazminattest1 = p.DataFrame(maasvetazminattest1.values)
    
    def pozisyonGorsellestir(self):
       
        plt.title("POZİSYONA GÖRE LİNEER REGRESYON GRAFİĞİ")
        
        plt.plot(maasvetazminattest1, color="blue", label="Gerçek Değerler")
        plt.plot(linetahP, color="red", lw=3, label= "Tahmini Değerler")
        plt.legend()
        plt.show()

        plt.title("POZİSYONA GÖRE GERÇEK DEĞERLER ")
        plt.pcolormesh(maasvetazminattest1)
        plt.show()

        plt.title("POZİSYONA GÖRE LİNEER REGRESYON TAHMİNİ DEĞERLERİ ")
        plt.pcolormesh(linetahP)
        plt.show()
    
    def pozisyonTahmin(self):    
        index=0
        tahmindf2=p.DataFrame(columns=pozisyonisimleri)
        for pozisyon in pozisyonisimleri:
            print(index, pozisyon)
            index+=1
        pozisyonx=input("Maaşını tahmin etmek istediğiniz futbolcunun pozisyonunu giriniz: ")
        pozisyonx = pozisyonx.upper() 
        tahmindf2.loc[0]=0
        tahmindf2[pozisyonx]=1
        tahminsonuc2 = lineer2.predict(tahmindf2)
        tahminsonuc2= p.DataFrame(tahminsonuc2, columns=["Tahmini Maaş","Tahmini Tazminat"]).iloc[0]
        print(tahminsonuc2)


#-----FUTBOLCULARIN ALDIKLARI MAAŞA GÖRE TAZMİNAT TAHMİNİ-----

    maasegitim,maastest,tazminategitim,tazminattest= tts(df.iloc[:,[0]],df.iloc[:,[1]], test_size=0.33)
    maasegitim = maasegitim.sort_index()
    maastest = maastest.sort_index()
    tazminategitim = tazminategitim.sort_index()
    tazminattest = tazminattest.sort_index()


    lineer3= lineerreg()
    lineer.fit(maasegitim,tazminategitim)
    tahminn = lineer.predict(maastest)
    #Lineer regresyon kullanılarak tahminleme ve görselleştirme yapıldı
    def LineerMaasTazminat(self):
        plt.title("LİNEER REGRESYONLA MAAŞA GÖRE TAZMİNAT TAHMİNİ")
        plt.scatter(maastest,tazminattest,color="red",label="Gerçek Değerler")
        plt.plot(maastest,tahminn, color="blue",label="Tahmini Değerler")
        plt.legend()
        plt.show()
    
        print("LİNEER REGRESYON ALGORİTMASININ DOĞRULUK YÜZDESİ: %{}".format(int(rkare(tazminattest,tahminn)*100)))
    #Karar ağacı kullanılarak tahminleme ve görselleştirme yapıldı
    def KararAgaciMaasTazminat(self):
        KararAgaci = DecisionTreeRegressor()
        KararAgaci.fit(maasegitim,tazminategitim)
        tahminAgac= KararAgaci.predict(maastest)
        plt.title("KARAR AĞACIYLA MAAŞA GÖRE TAZMİNAT TAHMİNİ")
        plt.scatter(maastest,tazminattest,color="red",label="Gerçek Değerler")
        plt.scatter(maastest,tahminAgac, color="blue",label="Tahmini Değerler")
        plt.legend()
        plt.show()
        
        plt.title("TAZMİNATLARIN GERÇEK DEĞERİ")
        plt.pcolormesh(tazminattest)
        plt.show()

        plt.title("KARAR AĞACINA GÖRE TAZMİNATLARIN TAHMİNİ DEĞERİ")
        tahminAgac = p.DataFrame(tahminAgac)
        plt.pcolormesh(tahminAgac)
        plt.show()
    
        print("KARAR AĞACI ALGORİTMASININ DOĞRULUK YÜZDESİ: %{}".format(int(rkare(tazminattest,tahminAgac)*100)))
    
    #Random forest kullanılarak tahminleme ve görselleştirme yapıldı
    def RandomForestMaasTazminat(self):
        RandomForest = RandomForestRegressor()
        RandomForest.fit(maasegitim,tazminategitim)
        tahminOrman = RandomForest.predict(maastest)
        plt.title("RANDOM FOREST İLE MAAŞA GÖRE TAZMİNAT TAHMİNİ")
        plt.scatter(maastest,tazminattest,color="red",label="Gerçek Değerler")
        plt.scatter(maastest,tahminOrman, color="blue",label="Tahmini Değerler")
        plt.legend()
        plt.show()

        plt.title("TAZMİNATLARIN GERÇEK DEĞERİ")
        plt.pcolormesh(tazminattest)
        plt.show()
        
        plt.title("RANDOM FOREST İLE TAZMİNATLARIN TAHMİNİ DEĞERİ")
        tahminOrman = p.DataFrame(tahminOrman)
        plt.pcolormesh(tahminOrman)
        plt.show()
    
        print("RANDOM FOREST ALGORİTMASININ DOĞRULUK YÜZDESİ: %{}".format(int(rkare(tazminattest,tahminOrman)*100)))

from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
veri = p.read_csv("datasetler/Sigortadataset.csv")
veri = p.DataFrame(veri)
veri["bmi"] = round(veri["bmi"])
veri.drop("region",axis=1, inplace=True)
class Classification():
       
    def LogisticRegression(self): #Lojistik regresyona göre sınıflandırma
        xtrain, xtest, ytrain, ytest = tts(veri[["age","bmi"]],veri["sex"],test_size=0.33)    
        LogisticR = LG()                
        LogisticR.fit(xtrain,ytrain)
        tahmin = LogisticR.predict(xtest)       
        ConfMatris=cm(tahmin,ytest)
        ConfMatris= p.DataFrame(data=ConfMatris,index=["Erkek Sayısı","Kadın Sayısı"],columns=["Erkek Tahmini"," Kadın Tahmin"])
        plt.title("LOJİSTİK REGRESYONA GÖRE SINIFLANDIRMANIN GÖRSELLEŞTİRİLMESİ\n")
        plt.pcolormesh(ConfMatris)
        plt.show()
        print("LOJİSTİK REGRESYON İÇİN KARMAŞIKLIK MATRİSİ")
        print(ConfMatris)
        dogru = ConfMatris.iloc[0,0] + ConfMatris.iloc[1,1]
        yanlis = ConfMatris.iloc[1,0] + ConfMatris.iloc[0,1]
        print("\nDoğru Sınıflandırma Sayısı: {}\nYanlış Sınıflandırma Sayısı: {}".format(dogru,yanlis))
    
        
    def SupportVectorMachine(self): #Destek vektör makinesi algoritmasına göre sınıflandırma
        xtrain, xtest, ytrain, ytest = tts(veri[["age","bmi"]],veri["sex"],test_size=0.33)
            
        supportvector = SVC(kernel="linear")
        supportvector.fit(xtrain,ytrain)
        tahmin = supportvector.predict(xtest) 
        ConfMatris=cm(tahmin,ytest)
        ConfMatris= p.DataFrame(data=ConfMatris,index=["Erkek Sayısı","Kadın Sayısı"],columns=["Erkek Tahmini"," Kadın Tahmin"])       
        plt.title("DESTEK VEKTÖR MAKİNESİNE GÖRE SINIFLANDIRMANIN GÖRSELLEŞTİRİLMESİ\n")
        plt.pcolormesh(ConfMatris)
        plt.show()
        print("DESTEK VEKTÖR MAKİNESİ İÇİN KARMAŞIKLIK MATRİSİ")
        print(ConfMatris)
        dogru = ConfMatris.iloc[0,0] + ConfMatris.iloc[1,1]
        yanlis = ConfMatris.iloc[1,0] + ConfMatris.iloc[0,1]
        print("\nDoğru Sınıflandırma Sayısı: {}\nYanlış Sınıflandırma Sayısı: {}".format(dogru,yanlis))
         
    def KNN_Classification(self): #K-Nearest Neighbour algoritmasına göre sınıflandırma
        xtrain, xtest, ytrain, ytest = tts(veri[["age","bmi"]],veri["sex"],test_size=0.33)
            
        knn = KNN(n_neighbors=3)
        knn.fit(xtrain,ytrain)
        tahmin = knn.predict(xtest) 
        ConfMatris=cm(tahmin,ytest)
        ConfMatris= p.DataFrame(data=ConfMatris,index=["Erkek Sayısı","Kadın Sayısı"],columns=["Erkek Tahmini"," Kadın Tahmin"])       
        plt.title("K-NN SINIFLANDIRMASINA GÖRE SINIFLANDIRMANIN GÖRSELLEŞTİRİLMESİ\n")
        plt.pcolormesh(ConfMatris)
        plt.show()
        print("K-NN SINIFLANDIRMASI İÇİN KARMAŞIKLIK MATRİSİ")
        print(ConfMatris)
        dogru = ConfMatris.iloc[0,0] + ConfMatris.iloc[1,1]
        yanlis = ConfMatris.iloc[1,0] + ConfMatris.iloc[0,1]
        print("\nDoğru Sınıflandırma Sayısı: {}\nYanlış Sınıflandırma Sayısı: {}".format(dogru,yanlis))
 
    def DecisionTree_Classifier(self): #Karar Agacına göre sınıflandırma
        xtrain, xtest, ytrain, ytest = tts(veri[["age","bmi"]],veri["sex"],test_size=0.33)
            
        dtc = DTC()
        dtc.fit(xtrain,ytrain)
        tahmin = dtc.predict(xtest) 
        ConfMatris=cm(tahmin,ytest)
        ConfMatris= p.DataFrame(data=ConfMatris,index=["Erkek Sayısı","Kadın Sayısı"],columns=["Erkek Tahmini"," Kadın Tahmin"])       
        plt.title("KARAR AĞACINA GÖRE SINIFLANDIRMANIN GÖRSELLEŞTİRİLMESİ\n")
        plt.pcolormesh(ConfMatris)
        plt.show()
        print("KARAR AĞACINA GÖRE SINIFLANDIRMA İÇİN KARMAŞIKLIK MATRİSİ")
        print(ConfMatris)
        dogru = ConfMatris.iloc[0,0] + ConfMatris.iloc[1,1]
        yanlis = ConfMatris.iloc[1,0] + ConfMatris.iloc[0,1]
        print("\nDoğru Sınıflandırma Sayısı: {}\nYanlış Sınıflandırma Sayısı: {}".format(dogru,yanlis))

    def RandomForest_Classifier(self): #Random foreste göre sınıflandırma
        xtrain, xtest, ytrain, ytest = tts(veri[["age","bmi"]],veri["sex"],test_size=0.33)
            
        rfc = RFC(n_estimators=10)
        rfc.fit(xtrain,ytrain)
        tahmin = rfc.predict(xtest) 
        ConfMatris=cm(tahmin,ytest)
        ConfMatris= p.DataFrame(data=ConfMatris,index=["Erkek Sayısı","Kadın Sayısı"],columns=["Erkek Tahmini"," Kadın Tahmin"])       
        plt.title("RANDOM FOREST ALGORİTMASINA GÖRE SINIFLANDIRMANIN GÖRSELLEŞTİRİLMESİ\n")
        plt.pcolormesh(ConfMatris)
        plt.show()
        print("RANDOM FOREST ALGORİTMASINA GÖRE SINIFLANDIRMA İÇİN KARMAŞIKLIK MATRİSİ")
        print(ConfMatris)
        dogru = ConfMatris.iloc[0,0] + ConfMatris.iloc[1,1]
        yanlis = ConfMatris.iloc[1,0] + ConfMatris.iloc[0,1]
        print("\nDoğru Sınıflandırma Sayısı: {}\nYanlış Sınıflandırma Sayısı: {}".format(dogru,yanlis))
#ARAYUZ TASARIMI    
import sys
from PyQt5 import QtWidgets as q
class Pencere(q.QWidget):
    def __init__(self):
        super().__init__()
        self.olustur()
        self.prediction = Prediction()
        self.classification = Classification()
              
    def olustur(self):
        baslik= q.QLabel("HANGİ TAHMİNLEME ALGORİTMASINI KULLANMAK İSTEDİĞİNİZİ SEÇİNİZ")
        
        a = q.QLabel("1.Lineer Regresyonla Takıma Göre Maaş-Tazminat Tahmini")
        b = q.QLabel("2.Lineer Regresyonla Pozisyona Göre Maaş-Tazminat Tahmini")
        c = q.QLabel("3.Lineer Regresyonla Takıma Göre Tahminlerin Görselleştirilmesi")
        d = q.QLabel("4.Lineer Regresyonla Pozisyona Göre Tahminlerin Görselleştirilmesi")
        e = q.QLabel("5.Lineer Regresyonla Maaşa Göre Tazminat Tahmini ve Görselleştirilmesi")
        f = q.QLabel("6.Karar Ağacıyla Maaşa Göre Tazminat Tahmini ve Görselleştirilmesi")
        g = q.QLabel("7.Random Forest İle Maaşa Göre Tazminat Tahmini ve Görselleştirilmesi")
        labelList=[baslik,a,b,c,d,e,f,g]
    
        butondizisi=[]
        for i in range(7):
            butondizisi.append(q.QPushButton("{}".format(i+1)))
                  
        hbox = q.QHBoxLayout()
        self.vbox= q.QVBoxLayout()
        for buton in butondizisi:           
            hbox.addWidget(buton)           
        for label in labelList:
            self.vbox.addWidget(label)            
        self.vbox.addStretch()
        self.vbox.addLayout(hbox)       
        self.setLayout(self.vbox)        
      
        butondizisi[0].clicked.connect(self.fonk1)
        butondizisi[1].clicked.connect(self.fonk2)
        butondizisi[2].clicked.connect(self.fonk3)
        butondizisi[3].clicked.connect(self.fonk4)
        butondizisi[4].clicked.connect(self.fonk5)
        butondizisi[5].clicked.connect(self.fonk6)
        butondizisi[6].clicked.connect(self.fonk7)        
        
        baslik= q.QLabel("HANGİ SINIFLANDIRMA ALGORİTMASINI KULLANMAK İSTEDİĞİNİZİ SEÇİNİZ")
        
        a = q.QLabel("1.Lojistik Regresyonla Vücut Endeksi, Yaş ve Birikime Göre Erkek-Kadın Sınıflandırması")
        b = q.QLabel("2.Destek Vektör Makinesi İle Vücut Endeksi, Yaş ve Birikime Göre Erkek-Kadın Sınıflandırması ")
        c = q.QLabel("3.K-Nearest Neighbor Algoritması İle Vücut Endeksi, Yaş ve Birikime Göre Erkek-Kadın Sınıflandırması")
        d = q.QLabel("4.Karar Ağacı İle Vücut Endeksi, Yaş ve Birikime Göre Erkek-Kadın Sınıflandırması")
        e = q.QLabel("5.Random Forest İle Vücut Endeksi, Yaş ve Birikime Göre Erkek-Kadın Sınıflandırması")
        
        labelList=[baslik,a,b,c,d,e]
    
        butondizisi2=[]
        for i in range(5):
            butondizisi2.append(q.QPushButton("{}".format(i+1)))
               
        hbox = q.QHBoxLayout()
        for buton in butondizisi2:
            
            hbox.addWidget(buton)           

        for label in labelList:
            self.vbox.addWidget(label)
            
        self.vbox.addStretch()
        self.vbox.addLayout(hbox)       
        self.setLayout(self.vbox)
        
        butondizisi2[0].clicked.connect(self.cfonk1)
        butondizisi2[1].clicked.connect(self.cfonk2)
        butondizisi2[2].clicked.connect(self.cfonk3)
        butondizisi2[3].clicked.connect(self.cfonk4)
        butondizisi2[4].clicked.connect(self.cfonk5)

        self.label = q.QLabel()
        self.vbox.addWidget(self.label)           
        self.label.setText("************ BUTONA TIKLADIKTAN SONRA KONSOLA BAKINIZ ************")
        hbox = q.QHBoxLayout()
        self.vbox.addLayout(hbox)        
        self.show()

    def fonk1(self):
        self.prediction.clubTahmin()           
    def fonk2(self):
        self.prediction.pozisyonTahmin()             
    def fonk3(self):
        self.prediction.clubGorsellestir()
    def fonk4(self):
        self.prediction.pozisyonGorsellestir()
    def fonk5(self):
        self.prediction.LineerMaasTazminat()
    def fonk6(self):
        self.prediction.KararAgaciMaasTazminat()
    def fonk7(self):
        self.prediction.RandomForestMaasTazminat()
        
    def cfonk1(self):
        self.classification.LogisticRegression()        
    def cfonk2(self):
        self.classification.SupportVectorMachine()
    def cfonk3(self):
        self.classification.KNN_Classification()
    def cfonk4(self):
        self.classification.DecisionTree_Classifier()
    def cfonk5(self):
        self.classification.RandomForest_Classifier()                                         
app = q.QApplication(sys.argv)
pencere = Pencere()
sys.exit(app.exec_())