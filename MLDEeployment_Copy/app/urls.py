from django.conf.urls import url
from app import views


urlpatterns = [
    url(r"test",views.testing,name='testing'),
    url(r"train",views.training,name='training'),
    url(r"integration",views.integration,name='integration'),
    url(r'',views.index,name='index')

]
