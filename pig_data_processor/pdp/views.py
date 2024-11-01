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
@require_POST
def userInfo(request):
    data = json.loads(request.body)
    userId = data.get('userId')
    userId_and_purchaseHistory = services.getUserInfoById(userId) # Object
    response_json = json.dumps(userId_and_purchaseHistory)
    return HttpResponse(response_json, status=200)

@csrf_exempt
@require_GET
def searchItem(request):
    searchTerm = request.GET.get('searchTerm')
    try:
        result = services.search_items(searchTerm)
        response_json = json.dumps(result)
        return HttpResponse(response_json, status=200)
    except Exception as ex:
        print(f"Error in searching item: {str(ex)}")
        return HttpResponse ('Error in search item', status=400)

@csrf_exempt
@require_POST
def recommendationByUser(request):
    data = json.loads(request.body)
    userId = data.get('userId')
    print(userId)    
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
    item = request.GET.get('searchTerm')
    print(itemId)
    print(userId)
    print('Search term------------------->',item)
    try:
        recommendedItems = services.recommend_similar_items(itemId, 7, userId,item)
        response_json = json.dumps(recommendedItems)
        return HttpResponse (response_json, status=200)
    except Exception as ex:
        print(f"Error getting item recommendation: {str(ex)}")
        return HttpResponse ('Error getting item recommendation', status=400)

# The following 3 apis are just for testing, users will not have access to them - they don't work now 
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

@csrf_exempt
@require_GET
def uploadImage(request):
    imageUrl = request.GET.get('imageUrl')
    userId = request.GET.get('userId')
    timestamp = '2024-10-30 17:19:11.871431'
    file_url = services.testUpload(imageUrl, userId,timestamp)
    response_json = json.dumps(file_url)
    return HttpResponse (response_json, status=200)


@csrf_exempt
@require_GET
def getPersonalizedItems(request):
    userId = request.GET.get('userId')
    result = services.getGeneratedImages(userId)
    response_json = json.dumps(result)
    return HttpResponse(response_json,status=200)