import json
from django.shortcuts import render
from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from . import services

def index(request):
    my_dict = {'insert_me': "From views.py"}
    return render(request,'pdp/index.html', context=my_dict)

@csrf_exempt
def pong(request):
    if request.method == 'GET':      
        return HttpResponse('pong')
    else:
        return HttpResponse('invalid request',status=400)

@csrf_exempt
@require_GET
def userInfo(request):
    userId = request.GET.get('userId')
    userId_and_purchaseHistory = services.getUserInfoById(userId) # Object
    response_json = json.dumps(userId_and_purchaseHistory)
    return HttpResponse(response_json, status=200)

@csrf_exempt
@require_GET
def recommendationByUser(request):
    userId = request.GET.get('userId')
    userId_and_purchaseHistory = services.getUserRecommendationByUserId(userId) # Object
    response_json = json.dumps(userId_and_purchaseHistory)
    return HttpResponse(response_json, status=200)

@csrf_exempt
@require_GET
def userRecommendation(request):
    try:
        services.generate_user_recommendation()
        return HttpResponse ('User recommendations successfully generated', status=200)
    except Exception as ex:
        print(f"Error generating user recommendation: {str(ex)}")
        return HttpResponse ('Did not generate user recommendation', status=400)

@csrf_exempt
@require_GET
def itemRecommendation(request):
    try:
        services.train_lstm_model()
        return HttpResponse ('Item embeddings successfully generated (LSTM)', status=200)
    except Exception as ex:
        print(f"Error training lstm: {str(ex)}")
        return HttpResponse ('Error training lstm', status=400)

@csrf_exempt
@require_GET
def recommendationByItem(request):
    itemId = request.GET.get('itemId')
    userId = request.GET.get('userId')
    item = request.GET.get('item')
    try:
        recommendedItems = services.recommend_similar_items(itemId, 5, userId,item)
        response_json = json.dumps(recommendedItems)
        return HttpResponse (response_json, status=200)
    except Exception as ex:
        print(f"Error getting item recommendation: {str(ex)}")
        return HttpResponse ('Error getting item recommendation', status=400)

# The following 2 apis are just for testing, users will not have access to them - they don't work now 
# To make them work, add the right params, see services methods for the right params
@csrf_exempt
@require_GET
def generatePrompt(request):
    prompt = services.generate_prompt_using_recs(request)
    return HttpResponse (status=200)

@csrf_exempt
@require_GET
def generateImage(request):
    imageUrl = services.generate_image('00004557432be3eeec63b4926113154e', 't-shirt')
    response_json = json.dumps(imageUrl)
    return HttpResponse (response_json, status=200)