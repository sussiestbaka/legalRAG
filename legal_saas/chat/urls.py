from django.urls import path
from .views import chat_complete

urlpatterns = [
    path('api/chat/', chat_complete, name='chat_complete'),
]
