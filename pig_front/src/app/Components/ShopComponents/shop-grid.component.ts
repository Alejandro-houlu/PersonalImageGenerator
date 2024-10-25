import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { NavbarComponent } from '../NavComponents/navbar.component';

@Component({
  selector: 'app-shop-grid',
  standalone: true,
  imports: [CommonModule, RouterLink,NavbarComponent],
  templateUrl: './shop-grid.component.html',
  styleUrl: './shop-grid.component.scss'
})
export class ShopGridComponent {

  product =
  
  [
    {
        "id":1,
        "image":"assets/images/shop/black-print-t-shirt.jpg",
        "tagText":"-40% Off",
        "name":"Black Print T-Shirt",
        "desRate":"$16.00",
        "rate":"$21.00",
        "tag":true
    },
    {
        "id":2,
        "image":"assets/images/shop/fashion-shoes-sneaker.jpg",
        "tagText":"New",
        "name":"Fashion Shoes Sneaker",
        "desRate":"$16.00",
        "rate":"$21.00"
    }
  ]

}
