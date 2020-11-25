import React from "react";
import Common from "./Common";
import Web from "../src/Images/svgimg.jpg";

const About = () => {
 return(
  <>
  <Common 
      name="Let's Create Something Unique with"
      imgscr={Web}
      visit="/Contact"
      btname="Contact Us"
  />
  </>
 )
} 
export default About;