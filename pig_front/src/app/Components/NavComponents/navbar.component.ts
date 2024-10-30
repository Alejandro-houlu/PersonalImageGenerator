import { CommonModule } from '@angular/common';
import { Component, Input, input } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { ClickOutsideModule } from 'ng-click-outside';
import * as feather from 'feather-icons';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, ClickOutsideModule],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.scss'
})
export class NavbarComponent {

  @Input() userId!: string
  toggleManu:boolean = false;
  manu:string = '';
  subManu: string = '';
  searchInput!:string;

  constructor(private router: Router){}

  ngAfterViewInit(){
    feather.replace()
  }
  toggleMenu(){
    this.toggleManu = !this.toggleManu;
  }
  openManu(item:string){
    this.subManu = item
  }

  //Search modal
  searchManu:boolean = false; 
  searchModal(){
    this.searchManu = !this.searchManu
  }
  closeSearchModal(){
    this.searchManu = false;
  }

  onSearchInput(event: Event){
    let input = event.target as HTMLInputElement
    console.info(input.value)
    this.searchInput = input.value
  }
  onSearchButtonClick(){
    console.info(this.searchInput)
    if(this.searchInput !== undefined && this.searchInput.trim() !== '' ){
      this.router.navigate(['/searchPage',this.searchInput,this.userId])
    }
  }
  
  //user-modal
  user:boolean = false;
  userModal(){
    this.user=!this.user
  }
  closeUserModal(){
    this.user = false;
  }

  //cart-modal
  cart:boolean = false;
  cartModal(){
    this.cart = !this.cart
  }
  closeCartModal(){
    this.cart = false; 
  }

}
