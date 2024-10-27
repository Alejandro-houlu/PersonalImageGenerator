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
    except Exception as ex:
        print(f"Error generating user recommendation: {str(ex)}")
        return HttpResponse ('Did not generate user recommendation', status=400)
    return HttpResponse(status=200)
