import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PersonalizedItemsDetailsComponent } from './personalized-items-details.component';

describe('PersonalizedItemsDetailsComponent', () => {
  let component: PersonalizedItemsDetailsComponent;
  let fixture: ComponentFixture<PersonalizedItemsDetailsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PersonalizedItemsDetailsComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(PersonalizedItemsDetailsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
