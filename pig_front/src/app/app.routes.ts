import { Routes } from '@angular/router';
import { ShopGridComponent } from './Components/ShopComponents/shop-grid.component';
import { LoginComponent } from './Components/AuthComponents/login.component';

export const routes: Routes = [
    {path:'', component:LoginComponent},
    {path:'shop/:userId',component:ShopGridComponent}
];
