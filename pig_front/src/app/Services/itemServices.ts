
import { Injectable } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { HttpClient } from "@angular/common/http";

@Injectable()
export class itemServices{

    constructor(private http:HttpClient){}

    searchItem(searchTerm:string){

        return firstValueFrom(this.http.get<any>(`/api/searchItem?searchTerm=${searchTerm}`))

    }

    showItem(itemId:string,userId:string,searchTerm:string){
        
        return firstValueFrom(this.http.get<any>(`/api/recommendationByItem?itemId=${itemId}&userId=${userId}&searchTerm=${searchTerm}`))
    }

    getPersonalizedItems(userId:string){
        return firstValueFrom(this.http.get<any>(`/api/getPersonalizedItems?userId=${userId}`))
    }




}