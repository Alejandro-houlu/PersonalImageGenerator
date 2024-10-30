import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, OnInit } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { NavbarComponent } from '../NavComponents/navbar.component';
import { itemServices } from '../../Services/itemServices';
import * as feather from 'feather-icons';


@Component({
  selector: 'app-personalized-items-details',
  standalone: true,
  imports: [CommonModule,RouterLink,NavbarComponent],
  templateUrl: './personalized-items-details.component.html',
  styleUrl: './personalized-items-details.component.scss',
  providers:[itemServices]
})
export class PersonalizedItemsDetailsComponent implements OnInit, AfterViewInit{

  constructor(private activatedRoute:ActivatedRoute,
    private itemSvc:itemServices
  ){}

  // Hard coded for aesthetics
  color=[
    'bg-red-600','bg-indigo-600','bg-emerald-600','bg-slate-900','bg-gray-400','bg-orange-600','bg-violet-600'
  ]
  brand =[
    'Alexander McQueen','Alexander Wang','Allegra K','AllSaints','Badgley Mischka','Baldinini'
  ]
  size = [
    'S', 'M', 'L', 'XL', '2XL', '3XL', '4XL'
  ]

  userId!:string
  items:any

  ngOnInit(): void {
    this.userId = this.activatedRoute.snapshot.params['userId']
    console.info(this.userId)

    this.itemSvc.getPersonalizedItems(this.userId)
      .then(res=>{
        console.info(res)
        this.items = res
        console.info(this.items)

      })


  }

  ngAfterViewInit(): void {
    feather.replace()
  }

}
