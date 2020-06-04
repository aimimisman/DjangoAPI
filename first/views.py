from django.shortcuts import render
from django.http import HttpResponse

from rest_framework.response import Response
#from rest_framework.request import Request
from rest_framework import status
from django.http import JsonResponse
from sklearn.externals import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create your views here.

from sklearn.externals import joblib


def index(request):
    temp={}
    temp['Gender'] = 0
    temp['Height'] = 0
    temp['Weight'] = 0
    context={'temp':temp}
    return render(request,'index.html',context)
    
def predictBMI(request):

    model = joblib.load('/Users/aimimisman/Desktop/BMIWebApp/models/BMI_model.pkl')
    fullinputdata = pd.read_csv('/Users/aimimisman/Desktop/BMIWebApp/first/500_Person_Gender_Height_Weight_Index.csv')
    fullinputdata['Gender'] = fullinputdata['Gender'].map({'Male': 0,'Female': 1})
    fullinputdata = fullinputdata.drop('Index', axis=1)
    sc = StandardScaler()

    if request.method == 'POST':
        temp={}
        temp['Gender']=int(request.POST.get('GenderVal'))
        temp['Height']=int(request.POST.get('HeightVal'))
        temp['Weight']=int(request.POST.get('WeightVal'))

        inputdata = pd.DataFrame([temp], columns=temp.keys())
        newdata = inputdata.append(fullinputdata)
        data_test = sc.fit_transform(newdata)
        prediction = model.predict(data_test)[0]

        result = pd.DataFrame([prediction], columns = ['Result'])
        result = result.replace({0 : 'Extremely Underweight', 2 : 'Underweight', 3 : 'Normal', 4 : 'Overweight', 5 : 'Obese'})
        
        context={'result':result}
        return render(request,'index.html',context)