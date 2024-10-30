import { Routes } from '@angular/router';
import { ShopGridComponent } from './Components/ShopComponents/shop-grid.component';
import { LoginComponent } from './Components/AuthComponents/login.component';
import { HistoryComponent } from './Components/HistoryComponents/history.component';
import { SearchComponent } from './Components/SearchComponents/search.component';
import { ItemDetailComponent } from './Components/ItemComponents/item-detail.component';
import { PersonalizedItemsDetailsComponent } from './Components/PigComponents/personalized-items-details.component';

export const routes: Routes = [
    {path:'', component:LoginComponent},
    {path:'shop/:userId',component:ShopGridComponent},
    {path:'history/:userId',component:HistoryComponent},
    {path:'searchPage/:searchTerm/:userId',component:SearchComponent},
    {path:'itemDetailSearched/:userId/:itemId/:searchTerm',component:ItemDetailComponent},
    {path:'personalImages/:userId',component:PersonalizedItemsDetailsComponent},
];
