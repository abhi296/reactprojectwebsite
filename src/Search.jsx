import React, { useState } from "react";
import SearchImg from "./SearchImg";

const Search = () => {
    const [image,setImage] = useState("");
    const Event = (event) =>{
    const data = event.target.value;
    setImage(data);
    };

 return(
  <>
  <div className="header_Input">
  <input className="input" type='text' placeholder="Search" 
  onChange={Event}
  value={image}></input>
  <SearchImg name={image}/>
  </div>
  </>
 )
} ;
export default Search;