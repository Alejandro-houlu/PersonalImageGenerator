import { AfterViewInit, Component, OnInit } from '@angular/core';
import { NavbarComponent } from "../NavComponents/navbar.component";
import { itemServices } from '../../Services/itemServices';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import * as feather from 'feather-icons';
import { Subscription } from 'rxjs';
import { NONE_TYPE } from '@angular/compiler';


@Component({
  selector: 'app-item-detail',
  standalone: true,
  imports: [CommonModule,RouterLink,NavbarComponent],
  templateUrl: './item-detail.component.html',
  styleUrl: './item-detail.component.scss',
  providers:[itemServices]
})
export class ItemDetailComponent implements OnInit, AfterViewInit {

  constructor(private activatedRoute:ActivatedRoute,
    private itemSvc: itemServices
  ){}

  activeTab:number = 1;
  itemId!:string
  searchTerm!:string
  userId!:string
  items:any
  private routeSub!: Subscription;
  tag: boolean = true
  tagText: string = 'New'
  desRate: string = '$16.00'
  rate: string = '$21.00'
  searchedItem:any


  ngOnInit(): void {
    this.routeSub = this.activatedRoute.paramMap.subscribe(params => {
      this.userId = params.get('userId') || '';
      this.itemId = params.get('itemId') || '';
      this.searchTerm = params.get('searchTerm') || '';
      console.info('user ID', this.userId)
      console.info('item ID:', this.itemId);
      console.info('Search Term:', this.searchTerm);
      
      this.itemSvc.showItem(this.itemId,this.userId,this.searchTerm)
        .then(res => {
          this.items = res
          this.searchedItem = this.items[0]
          this.items.slice(1)
          console.log(this.items)
          console.log(this.searchTerm)
        }).catch(err => console.error(err))
    });

  }

  ngAfterViewInit(): void {
    feather.replace()
   }

  onTabClick(index:number){
    this.activeTab=index;
  }

}
