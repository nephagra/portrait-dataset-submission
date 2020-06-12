$(document).ready(function(){
  var metadata = $.getJSON("js/portraits.json");
  console.log(metadata);
  var i = 0;
  //var tp = 2;


    $(function () {
        // setup an event handler to set the text when the range value is dragged (see event for input) or changed (see event for change)
        $('#YearRange1').on('input change', function () {
            $('#startyear').text("Start Year - " + $(this).val());
            if($('#YearRange1').val() > $('#YearRange2').val()){
                $('#searcherror').text("ERROR: Start Year greather than End Year.");
            }else{
                $('#searcherror').text("");
            }
        });
    });
    $(function () {
        // setup an event handler to set the text when the range value is dragged (see event for input) or changed (see event for change)
        $('#YearRange2').on('input change', function () {
            $('#endyear').text("End Year - " + $(this).val());
            if($('#YearRange1').val() > $('#YearRange2').val()){
                $('#searcherror').text("ERROR: Start Year greather than End Year.");
            }else{
                $('#searcherror').text("");
            }
        });
    });

    $('#m1').on({
        'click': function(){
            sessionStorage.setItem("tp", 1);
            window.location = 'data.html';
        }
    })

    $('#m2').on({
        'click': function(){
            sessionStorage.setItem("tp", 2);
            window.location = 'data.html';
        }
    })

    $('#m3').on({
        'click': function(){
            sessionStorage.setItem("tp", 3);
            window.location = 'data.html';
        }
    })

    $('#m4').on({
        'click': function(){
            sessionStorage.setItem("tp", 4);
            window.location = 'data.html';
        }
    })

    $('#m5').on({
        'click': function(){
            sessionStorage.setItem("tp", 5);
            window.location = 'data.html';
        }
    })

    $('#m6').on({
        'click': function(){
            sessionStorage.setItem("tp", 6);
            window.location = 'data.html';
        }
    })

    $('#m7').on({
        'click': function(){
            sessionStorage.setItem("tp", 7);
            window.location = 'data.html';
        }
    })

    $('#m8').on({
        'click': function(){
            sessionStorage.setItem("tp", 8);
            window.location = 'data.html';
        }
    })

    $('#m9').on({
        'click': function(){
            sessionStorage.setItem("tp", 9);
            window.location = 'data.html';
        }
    })

    $('#m10').on({
        'click': function(){
            sessionStorage.setItem("tp", 10);
            window.location = 'data.html';
        }
    })

    $('#m11').on({
        'click': function(){
            sessionStorage.setItem("tp", 11);
            window.location = 'data.html';
        }
    })

    $('#m12').on({
        'click': function(){
            sessionStorage.setItem("tp", 12);
            window.location = 'data.html';
        }
    })

    $(function () {
      if(sessionStorage.getItem("tp") == 1){
        $('#exampleRadios1').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 2){
        $('#exampleRadios2').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 3){
        $('#exampleRadios3').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 4){
        $('#exampleRadios4').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 5){
        $('#exampleRadios5').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 6){
        $('#exampleRadios6').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 7){
        $('#exampleRadios7').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 8){
        $('#exampleRadios8').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 9){
        $('#exampleRadios9').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 10){
        $('#exampleRadios10').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 11){
        $('#exampleRadios11').attr("checked", true);
      }
      else if(sessionStorage.getItem("tp") == 12){
        $('#exampleRadios12').attr("checked", true);
      }
    });

    
    $('#a1').on({
        'click': function(){
            sessionStorage.setItem("ar", 1);
            window.location = 'data.html';
        }
    })

    $('#a2').on({
        'click': function(){
            sessionStorage.setItem("ar", 2);
            window.location = 'data.html';
        }
    })

    $('#a3').on({
        'click': function(){
            sessionStorage.setItem("ar", 3);
            window.location = 'data.html';
        }
    })

    $('#a4').on({
        'click': function(){
            sessionStorage.setItem("ar", 4);
            window.location = 'data.html';
        }
    })

    $('#a5').on({
        'click': function(){
            sessionStorage.setItem("ar", 5);
            window.location = 'data.html';
        }
    })

    $('#a6').on({
        'click': function(){
            sessionStorage.setItem("ar", 6);
            window.location = 'data.html';
        }
    })

    $('#a7').on({
        'click': function(){
            sessionStorage.setItem("ar", 7);
            window.location = 'data.html';
        }
    })

    $('#a8').on({
        'click': function(){
            sessionStorage.setItem("ar", 8);
            window.location = 'data.html';
        }
    })

    $('#a9').on({
        'click': function(){
            sessionStorage.setItem("ar", 9);
            window.location = 'data.html';
        }
    })

    $('#a10').on({
        'click': function(){
            sessionStorage.setItem("ar", 10);
            window.location = 'data.html';
        }
    })

    $('#a11').on({
        'click': function(){
            sessionStorage.setItem("ar", 11);
            window.location = 'data.html';
        }
    })

    $('#a12').on({
        'click': function(){
            sessionStorage.setItem("ar", 12);
            window.location = 'data.html';
        }
    })

    $(function () {
      if(sessionStorage.getItem("ar") == 1){
        $('#artistsearch').attr("value", "Rembrandt");
      }
      else if(sessionStorage.getItem("ar") == 2){
        $('#artistsearch').attr("value", "Titian");
      }
      else if(sessionStorage.getItem("ar") == 3){
        $('#artistsearch').attr("value", "van Gogh");
      }
      else if(sessionStorage.getItem("ar") == 4){
        $('#artistsearch').attr("value", "Cezanne");
      }
      else if(sessionStorage.getItem("ar") == 5){
        $('#artistsearch').attr("value", "Modigliani");
      }
      else if(sessionStorage.getItem("ar") == 6){
        $('#artistsearch').attr("value", "Manet");
      }
      else if(sessionStorage.getItem("ar") == 7){
        $('#artistsearch').attr("value", "Raphael");
      }
      else if(sessionStorage.getItem("ar") == 8){
        $('#artistsearch').attr("value", "Renoir");
      }
      else if(sessionStorage.getItem("ar") == 9){
        $('#artistsearch').attr("value", "Tintoretto");
      }
      else if(sessionStorage.getItem("ar") == 10){
        $('#artistsearch').attr("value", "Degas");
      }
      else if(sessionStorage.getItem("ar") == 11){
        $('#artistsearch').attr("value", "Gauguin");
      }
      else if(sessionStorage.getItem("ar") == 12){
        $('#artistsearch').attr("value", "Frans Hals");
      }
    });

    $('#loadportrait').on({
        'click': function(){
            i += 1
            $('#reconin').attr('src','Images/' + metadata.responseJSON[i]['filename']);
        }
    })

});

function loadPortraitImg(timePeriod) {
  var timePerodsJSON = $.getJSON("js/time_periods.json");
  var timePeriodFiles = timePerodsJSON[timePeriod]
  return timePeriodFiles[Math.floor(Math.random() * timePeriodFiles.length)]
}


