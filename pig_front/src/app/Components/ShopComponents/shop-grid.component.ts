import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, OnInit } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { NavbarComponent } from '../NavComponents/navbar.component';
import { userServices } from '../../Services/userServices';
import * as feather from 'feather-icons';


@Component({
  selector: 'app-shop-grid',
  standalone: true,
  imports: [CommonModule, RouterLink,NavbarComponent],
  templateUrl: './shop-grid.component.html',
  styleUrl: './shop-grid.component.scss',
  providers: [userServices]

})
export class ShopGridComponent implements OnInit, AfterViewInit{

  constructor(private userSvc:userServices,
    private activatedRoute:ActivatedRoute
  ){}
  userRec: any;
  userId!: any;
  items:[]= [];
  tag: boolean = true
  tagText: string = 'New'
  desRate: string = '$16.00'
  rate: string = '$21.00'
  searchTerm:string =''



  ngOnInit(): void {

    this.userId = { "userId" :this.activatedRoute.snapshot.params['userId']}
    console.info(this.userId)

    this.userSvc.login(this.userId)
      .then(res=>{
        console.info(res)
        this.userRec = res
        this.items = res.item_info
        console.info(this.items)
      }).catch(err => {
        console.error(err)
      })

  }
  

  ngAfterViewInit(): void {
    feather.replace()

  }

}
