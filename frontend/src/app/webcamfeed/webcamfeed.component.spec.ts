import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { WebcamfeedComponent } from './webcamfeed.component';

describe('WebcamfeedComponent', () => {
  let component: WebcamfeedComponent;
  let fixture: ComponentFixture<WebcamfeedComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ WebcamfeedComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(WebcamfeedComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
