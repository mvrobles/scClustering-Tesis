package com.example;

import org.springframework.web.bind.annotation.RestController;

import lombok.extern.slf4j.Slf4j;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.HashMap;
import java.util.Map;

@RestController
@Slf4j
public class ControladorREST {

    @GetMapping("/")
    public String comienzo(){
        log.info("Estoy ejecutando el controlador REST");
        log.debug("Mas información");
        return "Hola Mundo en Spring";
    }

    @GetMapping("/api/greet")
    public ResponseEntity<Map<String,String>> greetUser(String name, Integer age) {
        Map<String,String> response = new HashMap<>();
        response.put("message", "Hola, " + name);

        if (age != null){
            response.put("age", "Tienes " + age + " años");
        }
        return ResponseEntity.ok(response);
    }
    
}
