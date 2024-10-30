import { Injectable } from "@angular/core";
import { BehaviorSubject, firstValueFrom } from "rxjs";
import { HttpClient } from "@angular/common/http";

@Injectable()
export class userServices{

    constructor(private http:HttpClient){}

    login(userId:String){

        return firstValueFrom(
            this.http.post<any>('api/recommendationByUser',userId)
        ).then(res =>{
            return res
        })
    }

    getUserHistory(userId:String){
        let userIdObj = {"userId": userId}
        return firstValueFrom(
            this.http.post<any>('api/userInfo',userIdObj)
        ).then(res => res)
        .catch(err => console.info(err))
    }

}