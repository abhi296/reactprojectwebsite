import React from "react";

const Contact = () => {
 return(
  <>
  <div className="my-5">
    <h1 className="text-center">Contact Us</h1>
  </div>
  <div className="container contact_div">
    <div className="row">
      <div className="col-md-6 col-10 mx-auto">
        <form>
  <div class="mb-3">
  <label for="exampleFormControlInput1" class="form-label">First Name:</label>
  <input type="text" 
  class="form-control" 
  id="exampleFormControlInput1" 
  placeholder="Enter Your Full Name"/>
  </div>
  <div class="mb-3">
  <label for="exampleFormControlInput1" class="form-label">Last Name:</label>
  <input type="text" 
  class="form-control" 
  id="exampleFormControlInput1" 
  placeholder="Enter Your Last Name"/>
  </div>
  <div class="mb-3">
  <label for="exampleFormControlInput1" class="form-label">Mobile Number:</label>
  <input type="number" 
  class="form-control" 
  id="exampleFormControlInput1" 
  placeholder="Enter Your Mobile Number"/>
  </div>
  <div class="mb-3">
  <label for="exampleFormControlInput1" class="form-label">Email Id:</label>
  <input type="text" 
  class="form-control" 
  id="exampleFormControlInput1" 
  placeholder="Enter Your Email Id"/>
  </div>
  <div class="mb-3">
  <label for="exampleFormControlTextarea1" 
  class="form-label">Comments</label>
  <textarea class="form-control" 
  id="exampleFormControlTextarea1" 
  rows="3"></textarea>
</div>
<div class="col-12">
    <button class="btn btn-dark" type="submit">Submit</button>
  </div>
        </form>
      </div>
    </div>
  </div> 
   </>
 )
} 
export default Contact;