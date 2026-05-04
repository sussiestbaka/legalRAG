from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import ChatSession, ChatMessage
from .rag_loader import vectordb, doRAG
from django.contrib.auth.models import User

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_complete(request):
    user = request.user
    data = request.data
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', None)

    if not user_message:
        return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Get or create chat session
    if session_id:
        try:
            session = ChatSession.objects.get(id=session_id, user=user)
        except ChatSession.DoesNotExist:
            return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    else:
        session = ChatSession.objects.create(user=user, title=user_message[:50])

    # Build conversation history for the agent (last 4 exchanges)
    history = []
    for msg in session.messages.order_by('-timestamp')[:8]:  # 4 exchanges = 8 messages
        history.append({"role": msg.role, "content": msg.content})
    history.reverse()   # oldest first

    # Call your existing doRAG function
    answer = doRAG(user_message, vectordb, history=history)

    # Save user message and assistant response
    user_msg = ChatMessage.objects.create(session=session, role='user', content=user_message)
    assistant_msg = ChatMessage.objects.create(session=session, role='assistant', content=answer, sources=[])

    return Response({
        'answer': answer,
        'session_id': session.id,
        'message_id': assistant_msg.id,
    })