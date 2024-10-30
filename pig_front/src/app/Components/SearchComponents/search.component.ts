import { AfterViewInit, Component, OnInit } from '@angular/core';
import { itemServices } from '../../Services/itemServices';
import { CommonModule } from '@angular/common';
import { NavbarComponent } from '../NavComponents/navbar.component';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import * as feather from 'feather-icons';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule,NavbarComponent,RouterLink],
  templateUrl: './search.component.html',
  styleUrl: './search.component.scss',
  providers:[itemServices]
})
export class SearchComponent implements OnInit, AfterViewInit {

  constructor(private activatedRoute:ActivatedRoute,
    private itemSvc: itemServices
  ){}
  tag: boolean = true
  tagText: string = 'New'
  desRate: string = '$16.00'
  rate: string = '$21.00'
  userId!:string
  searchTerm!:string
  private routeSub!: Subscription;
  items:[]=[]

  ngOnInit(): void {

    this.routeSub = this.activatedRoute.paramMap.subscribe(params => {
      this.userId = params.get('userId') || '';
      this.searchTerm = params.get('searchTerm') || '';
      console.info('User ID:', this.userId);
      console.info('Search Term:', this.searchTerm);
      
      this.itemSvc.searchItem(this.searchTerm)
        .then(res => {
          this.items = res.result
          console.info(this.items)
        }).catch(err => console.error(err))

    });
  }
  ngAfterViewInit(): void {
    feather.replace()
  }

}
