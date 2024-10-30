import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, OnInit } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { NavbarComponent } from '../NavComponents/navbar.component';
import { userServices } from '../../Services/userServices';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, RouterLink, NavbarComponent],
  templateUrl: './history.component.html',
  styleUrl: './history.component.scss',
  providers:[userServices]
})
export class HistoryComponent implements OnInit, AfterViewInit{

  constructor(private activatedRoute:ActivatedRoute,
    private userSvc:userServices
  ){}
  
  userId!:string
  items: []=[]
  tag: boolean = true
  tagText: string = 'Fea'
  desRate: string = '$16.00'
  rate: string = '$21.00'

  ngOnInit(): void {
    
    this.userId = this.activatedRoute.snapshot.params['userId']
    this.userSvc.getUserHistory(this.userId)
      .then(res =>{
        this.items = res.result
        console.info(this.items)
      })


  }
  ngAfterViewInit(): void {
      console.info(this.userId)
      
  }
 
}
