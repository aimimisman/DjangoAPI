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

model = joblib.load('/Users/aimimisman/Desktop/BMIWebApp/models/BMI_model.pkl')
sc = StandardScaler()

def index(request):
    temp={}
    temp['Gender'] = 0
    temp['Height'] = 0
    temp['Weight'] = 0
    context={'temp':temp}
    return render(request,'index.html',context)
    #return HttpResponse({'a':1})
    
def predictBMI(request):
    if request.method == 'POST':
        temp={}
        temp['Gender']=int(request.POST.get('GenderVal'))
        temp['Height']=int(request.POST.get('HeightVal'))
        temp['Weight']=int(request.POST.get('WeightVal'))

        #testData=pd.DataFrame({'x':temp}).transpose()
        testData = pd.DataFrame([temp], columns=temp.keys())
        print(testData)
        data_test = sc.fit_transform(testData)
        scoreval=model.predict(data_test)[0]
        context={'scoreval':scoreval,'temp':temp}
        return render(request,'index.html',context)