import React from "react";
import { NavLink } from "react-router-dom";
import Web from "../src/Images/svgimg3.jpg";
import Common from "./Common";

const Home = () => {
 return(
  <>
  <Common
    name="Grow your Business with"
      imgscr={Web}
      visit="/Service"
      btname="Get Started"
  />
  </>
 );
};
export default Home;