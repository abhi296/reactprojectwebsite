import React from "react";

const SearchImg = (props) => {
const image = `https://source.unsplash.com/700x500/?${props.name}`;
 return(
  <>
  <img src ={image} alt="wallpaper"/>
  </>
 )
} 
export default SearchImg;
