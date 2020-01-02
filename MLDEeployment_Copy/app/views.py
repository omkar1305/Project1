from django.shortcuts import render, HttpResponse
from app import code
import pandas as pd
# Create your views here.

data_frame=pd.DataFrame()
def index(request):
    return render(request,'home/index.htm')

def training(request):
    if request.method=='POST':
        if(request.FILES['document']):
            #print(request.POST['train_upload'])
            uploaded_file=request.FILES['document']
            df=code.TrainModel(uploaded_file)
    return render(request,'home/training.htm')


def testing(request):
    if request.method=='POST':
        if(request.FILES['document']):
            uploaded_file=request.FILES['document']
            code.uploadData(uploaded_file)
    return render(request,'home/testing.htm')

def integration(request):
    if request.method =='POST':
        code.rallyData()

    return render(request, 'home/integration.html')