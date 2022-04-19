1.Kaggle比賽簡介：
![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/Description.png)

2.檔案介紹：

"train.sh"是腳本，可以執行訓練以及測試後生成csv檔案到資料夾<csv_data>

"108360150_house_pred.py"是訓練測試的py檔

"108360150_house_pred.ipynb"是訓練測試的ipynb檔

"tabular.ipynb"是用AutoGluon訓練測試的ipynb檔

"IMG"中是報告需要的所有圖片

"csv_data"中儲存了所有csv檔案

"house_pred_keras.csv"是用keras測試後的結果

"house_pred_tabular.csv"是用AutoGluon測試後的結果



3.程式：

我使用 jupyter notebook 編譯程式
![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/jupyter1.jpg)
![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/jupyter2.jpg)

程式：
        TRAIN PART：

        #colab中可以用下面兩行式子讀取文件，但jupyter中不需要，故作註記
        #from google.colab import files
        #uploaded = files.upload()
       
        #import 所有所需要的API
        import pandas as pd
        from keras.models import Sequential
        from keras import layers
        from keras import optimizers
        from keras.layers import Dense
        from keras.callbacks import ModelCheckpoint
        
        #讀取train-v3.csv，valid-v3.csv，test-v3.csv
        #將train和valid的id和price特徵去掉，並且將price另存至Y_train Y_valid
        data_train = pd.read_csv('train-v3.csv')
        X_train = data_train.drop(['price','id'],axis=1)
        Y_train = data_train['price'].values

        data_valid=pd.read_csv('valid-v3.csv')
        X_valid = data_valid.drop(['price','id'],axis=1)
        Y_valid = data_valid['price'].values

        data_test = pd.read_csv('test-v3.csv')
        X_test = data_test.drop(['id'],axis=1)

        #對所有數據做標準化(數據中沒有文字)
        mean,std=X_train.mean(axis=0,),X_train.std(axis=0)
        X_train=(X_train-mean)/std
        X_valid=(X_valid-mean)/std
        X_test=(X_test-mean)/std


        #利用keras建立模型，由於樣本數量較少，所以用較少的神經元和層數
        #用adam作為模型的優化器
        model =Sequential()
        model.add(Dense(128,input_dim=X_train.shape[1],kernel_initializer='random_normal',activation='relu'))
        model.add(Dense(64,kernel_initializer='normal',activation='relu'))
        model.add(Dense(32,kernel_initializer='normal',activation='relu'))
        model.add(Dense(1))
        adam=keras.optimizers.Adam(learning_rate=0.0081,beta_1=0.9, beta_2= 0.99, epsilon= None, decay=0.0, amsgrad= False)
        model.compile(loss='MAE',optimizer=adam)
        
        history = model.fit(X_train,Y_train,validation_data=(X_valid,Y_valid),
                        
                        epochs=250,batch_size=128,verbose=1) #epochs250次
       
         #計算loss，並畫出訓練過程
        losses=pd.DataFrame(model.history.history)
        losses.plot()
        model.summary()
        #訓練結果如圖所示：
![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/figure1.png)
![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/figure2.png)
         
#adam優化器介紹：https://www.jianshu.com/p/aebcaf8af76e
       
  TEST  PART：
                
                #將test數據丟到訓練好的模型當中，把預測好的數據存在house_predict.csv中
                        pred=model.predict(X_test)
                        with open('csv_data/house_pred_keras.csv','w')as f:
                                f.write('id,price\n')
                                for i in range(len(pred)):
                                        f.write(str(i+1)+','+str(float(pred[i]))+'\n')

Kaggle上傳：
![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/kaggle_sub.png)



4.心得：
由於剛剛接觸機器學習和Python,有時候了解了機器學習的觀念卻很難用程式去寫出來。
大家的模型用的都差不多，能夠優化的是對數據的處理和學習參數的調整，在adam中我去調整了learing rate，類似在SGD（Gradient Descent）中，學習率不能太小也不能太大，需要我們作調整。
我嘗試對原始數據進行調整，譬如將一些對房價影響不大的特徵去處，但是可能是我程式寫的有問題，去處後會一直有error，所以我放棄了對數據的調整。
雖然我沒有對數據進行調整，但是經過訓練後的模型誤差也能落在六七萬左右，跟大家的差不多，應該是強大的Adam對我很醜陋的參數進行了優化。
現在不像以前所有的程式都要自己去寫，網路上有很多前輩寫的強大的模組，譬如我在網路上找到一個強大的工具包AutoGluon，可以做出不錯的預測結果，程式如下：

  AutoGluon House predict:

![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/AutoGluon1.png)

        #訓練
        from autogluon.tabular import TabularDataset, TabularPredictor
        train_data = TabularDataset('train-v3.csv')
        id , label ='id','price'
        Predictor=TabularPredictor(label=label).fit(train_data.drop(columns=[id]))
       
        #預測
        import pandas as pd
        test_data = TabularDataset('test-v3.csv')
        preds = Predictor.predict(test_data.drop(columns=[id]))
        submission1 = pd.DataFrame({id:test_data[id],label:preds})
        submission1.to_csv('tabular_pred.csv',index=False)

 Kaggle上傳：
![image](https://github.com/MachineLearningNTUT/regression-T108360150/blob/main/IMG/tabular2.png)
雖然這個工具很強大，我們也應該對其進行人工的數據預處理來達到更好的效果，不過我現在的能力還沒辦法實現，希望能在幾周後再來嘗試，得到更好的成績。

當然也不能過於依賴這個強大的工具包，還是要從機器學習的最底層的觀念去學習，為將來研發新技術打好基礎。