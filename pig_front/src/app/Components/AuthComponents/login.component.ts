import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { Router, RouterLink} from '@angular/router';
import { userServices } from '../../Services/userServices';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, RouterLink, ReactiveFormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
  providers: [userServices]
})
export class LoginComponent implements OnInit {
  
  date:any = ''
  loginForm!:FormGroup

  constructor(private formBuilder:FormBuilder,private userSvc:userServices,private router:Router){}

  ngOnInit(){
    this.date = new Date().getFullYear();
    this.loginForm = this.formBuilder.group({
      userId: this.formBuilder.control<String>('')
    })

  }

  processLoginForm(){
    let userId = this.loginForm.value
    console.info(userId)

    this.router.navigate(['/shop',userId.userId])

  }

}


