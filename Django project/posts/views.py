import re
from django.shortcuts import redirect, render
from .models import searchdata
from django.contrib.auth.models import User, auth
from django.contrib import messages
 
# Create your views here.
def index(request):
    return render(request, 'index.html')

def home(request):
    return render(request,'index.html')
    
def newpage(request):
    return render(request,'newpage.html')

def team(request):
    return render(request,'team.html')

movielist = {'Action&Adventure':'The MATRIX','Animation':'Rise of the Guardians','Art house&International':'Parasite','Comedy':'Bad boys','Classics':'The Wizard of Oz','Drama':'Forrest Gump','Documentory':'The Shawshank Redemption','Kids&family':'Back to the future','Horror':'It','Mystery&Suspense':'Get out','Music&Performing arts':'Les Misérables','Science fiction&Fantasy':'Dune','Romance':'One day','Special interest':'SHAUN OF THE DEAD','Sport&Fitness':'Coach Carter','Television':'Peaky blinders','Western':'The Revenant'}
def check(request):
    if request.method == 'POST':
        movietypes = request.POST.getlist('movie')
        movietypesdata = str(movietypes[0]).replace('&','__').replace(' ','_').lower()
        moviedata = searchdata('https://www.rottentomatoes.com/top/bestofrt/top_100_{}_movies/'.format(movietypesdata)) 
        return render(request,'next.html',{'movie':moviedata[43:53],'movietypes':movietypes[0],'personalrec':movielist[movietypes[0]]}) 
    else:
        print('nothing is checked')
        return render(request,'next.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']

        if password == password2:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'Email already used')
                return redirect('register')
            elif User.objects.filter(username=username).exists():
                messages.info(request, 'Username already used')
                return redirect('register')
            else:
                user = User.objects.create_user(username=username,email=email,password=password)
                user.save()
                return redirect('login')
        else:
            messages.info(request, 'Password are not the same')
            return redirect('register')
    else:
        return render(request,'register.html')  

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username,password=password)
        
        if user is not None:
            auth.login(request, user)
            return redirect('/')
        else:
            messages.info(request, 'Credentials invalid')
            return redirect('login')
    else:
        return render(request, 'login.html')

def logout(request):
    auth.logout(request)
    return redirect('/')
